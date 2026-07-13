// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Demo LLM proxy.
//!
//! HTTP serving and API translation come entirely from switchyard's crates
//! (`switchyard-server` axum router + `switchyard-translation`, reached through
//! the `Profile` runtime). ALL routing is implemented with `libsy`: each request
//! is routed by libsy's LLM-classifier ([`libsy::llm_class`]), which calls a
//! classifier model to score the request and then routes to a strong or weak
//! model. libsy's targets make their model calls through switchyard's
//! OpenAI-compatible backend.
//!
//! The proxy serves all three inbound APIs — OpenAI (`/v1/chat/completions`),
//! Anthropic (`/v1/messages`), and Responses (`/v1/responses`) — and switchyard
//! translates the upstream response back to whichever the caller used.
//!
//! Env:
//!   ANTHROPIC_API_KEY     upstream bearer key (e.g. `$INFERENCE_HUB_SY_API_KEY`)
//!   LIBSY_PROXY_BASE_URL  upstream base url (default https://inference-api.nvidia.com/v1)
//!   LIBSY_PROXY_ADDR      listen address (default 127.0.0.1:4000)
//!
//! Run:
//!   ANTHROPIC_API_KEY=$INFERENCE_HUB_SY_API_KEY cargo run -p libsy-proxy

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use libsy::llm_class::{ClassifierDecision, LlmClassifierOrchAlgo};
use libsy::{
    Algorithm, Context, Decision, LlmClient, LlmContentBlock, LlmMessage, LlmRequest, LlmResponse,
    LlmResponseOutput, LlmRole, LlmTarget, LlmTargetSet, Request as LibsyRequest,
    Response as LibsyResponse, RoutedRequest,
};

use switchyard_components::OpenAiPassthroughBackend;
use switchyard_components_v2::{Profile, ProfileInput, ProfileResponse, RoutingMetadata};
use switchyard_core::{
    ChatRequest, ChatResponse, EndpointConfig, LlmBackend, ModelId, ProxyContext, Result,
    SwitchyardError,
};
use switchyard_server::{build_switchyard_router, ProfileRegistry, ServerState};
use tokio::net::TcpListener;

// Routing configuration (per the demo's inference-hub models).
const CLASSIFIER_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const STRONG_MODEL: &str = "aws/anthropic/bedrock-claude-opus-4-7";
const WEAK_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const CLASSIFIER_THRESHOLD: f64 = 0.5;

const DEFAULT_BASE_URL: &str = "https://inference-api.nvidia.com/v1";
const DEFAULT_ADDR: &str = "127.0.0.1:4000";
/// Model id callers address to reach this proxy (routing picks the real model).
const PROFILE_MODEL_ID: &str = "libsy-classifier";

/// A libsy [`LlmClient`] whose model call is performed by switchyard's
/// OpenAI-compatible backend — so libsy owns routing while switchyard owns the
/// HTTP transport. One instance is shared by every routing target.
struct SwitchyardBackendClient {
    backend: Arc<OpenAiPassthroughBackend>,
}

#[async_trait]
impl LlmClient for SwitchyardBackendClient {
    async fn call(
        &self,
        routed: RoutedRequest,
    ) -> std::result::Result<LibsyResponse, Box<dyn std::error::Error + Send + Sync>> {
        let model = routed.decision.selected_model().to_string();
        let prompt = routed
            .request
            .llm_request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .chain(
                routed
                    .request
                    .llm_request
                    .messages
                    .iter()
                    .flat_map(|message| message.content.iter()),
            )
            .filter_map(|block| match block {
                LlmContentBlock::Text { text }
                | LlmContentBlock::Refusal { text }
                | LlmContentBlock::Reasoning { text, .. } => Some(text.as_str()),
                LlmContentBlock::Unknown { raw, .. } => raw.as_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        // Build a single-shot OpenAI chat request for the chosen model.
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false,
        });
        let chat_request = ChatRequest::openai_chat(body);

        let mut ctx = ProxyContext::new();
        let response = self
            .backend
            .call(&mut ctx, &chat_request)
            .await
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

        let raw = response.body().cloned().unwrap_or(Value::Null);
        let completion = completion_text(&raw).unwrap_or_default();
        Ok(LibsyResponse {
            llm_response: LlmResponse {
                outputs: vec![LlmResponseOutput {
                    role: LlmRole::Assistant,
                    content: vec![LlmContentBlock::Text { text: completion }],
                    stop_reason: None,
                }],
                ..LlmResponse::default()
            },
            metadata: None,
        })
    }
}

/// A switchyard [`Profile`] that routes every request through libsy's LLM
/// classifier. switchyard's router hands us the inbound request and translates
/// our response back to the caller's format; we only do routing + one upstream call.
struct LibsyClassifierProfile {
    orchestrator: Arc<dyn Algorithm>,
}

#[async_trait]
impl Profile for LibsyClassifierProfile {
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        // Pull the user's prompt out of whatever inbound wire format we got.
        let prompt = extract_prompt(input.request.body())
            .ok_or_else(|| SwitchyardError::InvalidRequest("no user prompt in request".into()))?;

        let orch_request = LibsyRequest {
            llm_request: LlmRequest {
                model: Some(PROFILE_MODEL_ID.to_string()),
                messages: vec![LlmMessage::text(LlmRole::User, prompt)],
                ..LlmRequest::default()
            },
            raw_request: Some(input.request.body().clone()),
            metadata: None,
        };

        // ALL routing happens here, in libsy: the classifier scores the request
        // (one model call) and routes to the strong/weak model (a second call).
        // libsy's targets perform those calls via the switchyard backend.
        let (trace, response) = self
            .orchestrator
            .clone()
            .run(Context::default(), orch_request)
            .await
            .map_err(|e| SwitchyardError::Other(e.to_string()))?;

        // Return an OpenAI chat-completion body; switchyard reconciles it against
        // the caller's inbound format.
        let content = response
            .llm_response
            .outputs
            .iter()
            .flat_map(|output| output.content.iter())
            .filter_map(|block| match block {
                LlmContentBlock::Text { text }
                | LlmContentBlock::Refusal { text }
                | LlmContentBlock::Reasoning { text, .. } => Some(text.as_str()),
                LlmContentBlock::Unknown { raw, .. } => raw.as_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let body = json!({
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop",
            }],
        });
        let chat_response = ChatResponse::openai_completion(body);
        Ok(ProfileResponse::with_routing_metadata(
            chat_response,
            routing_metadata(&trace),
        ))
    }
}

/// Extract the user prompt from an inbound body, handling OpenAI / Anthropic
/// (`messages[]`) and Responses (`input`) shapes.
fn extract_prompt(body: &Value) -> Option<String> {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        if let Some(user) = messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        {
            if let Some(text) = user.get("content").and_then(content_to_text) {
                return Some(text);
            }
        }
    }
    // Responses API: `input` may be a string or an array of content items.
    body.get("input").and_then(content_to_text)
}

/// Flatten a message `content` (a string or an array of `{ text }` blocks) to text.
fn content_to_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Read the assistant text out of an OpenAI chat-completion body.
fn completion_text(body: &Value) -> Option<String> {
    body.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Surface libsy's routing decision as `x-model-router-*` response headers.
fn routing_metadata(trace: &[Arc<dyn Decision>]) -> RoutingMetadata {
    // The classifier trace is [classify, route]; the routed decision is last.
    let decision = trace.last();
    let classifier = decision.and_then(|d| d.as_any().downcast_ref::<ClassifierDecision>());
    RoutingMetadata {
        selected_model: decision.map(|d| d.selected_model().to_string()),
        selected_tier: classifier.and_then(|c| c.tier.map(|t| t.as_str().to_string())),
        confidence: classifier.and_then(|c| c.score),
        router_version: Some("libsy-classifier".to_string()),
        tolerance: None,
        rationale: decision.and_then(|d| d.reasoning().map(str::to_string)),
    }
}

fn build_orchestrator() -> Result<Arc<dyn Algorithm>> {
    let base_url =
        std::env::var("LIBSY_PROXY_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        SwitchyardError::InvalidConfig(
            "ANTHROPIC_API_KEY must be set to upstream bearer key".into(),
        )
    })?;
    let endpoint = EndpointConfig {
        base_url: Some(base_url),
        api_key: Some(api_key),
        timeout_secs: Some(120.0),
    };
    let backend = Arc::new(OpenAiPassthroughBackend::new(endpoint)?);
    let client = Arc::new(SwitchyardBackendClient { backend }) as Arc<dyn LlmClient>;

    // One target per model id; all backed by the same upstream client. The
    // provider model id is the semantic target name in this demo.
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone()),
    };
    let targets = LlmTargetSet::new(vec![
        target(CLASSIFIER_MODEL),
        target(STRONG_MODEL),
        target(WEAK_MODEL),
    ]);

    Ok(Arc::new(LlmClassifierOrchAlgo::new(
        CLASSIFIER_MODEL,
        STRONG_MODEL,
        WEAK_MODEL,
        CLASSIFIER_THRESHOLD,
        targets,
    )))
}

#[tokio::main]
async fn main() -> Result<()> {
    let orchestrator = build_orchestrator()?;
    let profile = Arc::new(LibsyClassifierProfile { orchestrator }) as Arc<dyn Profile>;

    let registry = ProfileRegistry::from_profiles([(
        ModelId::new(PROFILE_MODEL_ID)?,
        profile,
        PROFILE_MODEL_ID.to_string(),
    )])?;
    let state = ServerState::new(registry);

    let requested_addr: SocketAddr = std::env::var("LIBSY_PROXY_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_string())
        .parse()
        .map_err(|e: std::net::AddrParseError| SwitchyardError::InvalidConfig(e.to_string()))?;
    let listener = TcpListener::bind(requested_addr)
        .await
        .map_err(|e| SwitchyardError::Other(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| SwitchyardError::Other(e.to_string()))?;

    println!("libsy-proxy listening on http://{addr}");
    println!("  routing (libsy classifier): classifier={CLASSIFIER_MODEL}");
    println!("                              strong={STRONG_MODEL}");
    println!("                              weak={WEAK_MODEL}");
    println!(
        "  send model \"{PROFILE_MODEL_ID}\" to /v1/chat/completions, /v1/messages, or /v1/responses"
    );

    axum::serve(listener, build_switchyard_router(state))
        .await
        .map_err(|e| SwitchyardError::Other(e.to_string()))
}
