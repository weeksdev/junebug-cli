//! Provider-neutral streaming model contract with OpenAI-compatible and
//! Anthropic Messages REST adapters.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{Value, json};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound for one whole streamed turn. The blocking reqwest client
/// otherwise defaults to a 30-second total request timeout, which aborts
/// any stream that takes longer than that to finish.
const TURN_TIMEOUT: Duration = Duration::from_mins(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ModelTurn {
    pub text_deltas: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    pub assistant_message: Value,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub trait ModelProvider {
    fn name(&self) -> &'static str;
    /// Stream one model turn. Implementations must check `cancel` while
    /// streaming and return the partial turn (with tool calls discarded)
    /// once it is set.
    ///
    /// # Errors
    ///
    /// Returns an error for transport, protocol, or provider failures.
    fn stream_turn(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<ModelTurn, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAi,
    OpenRouter,
    DeepSeek,
    Anthropic,
    Ollama,
    LocalOpenAi,
}

impl ProviderKind {
    /// # Errors
    ///
    /// Returns an error when `value` is not a supported provider identifier.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "openai" => Ok(Self::OpenAi),
            "openrouter" => Ok(Self::OpenRouter),
            "deepseek" => Ok(Self::DeepSeek),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "ollama" | "local" => Ok(Self::Ollama),
            "local-openai" | "openai-local" | "lmstudio" | "vllm" => Ok(Self::LocalOpenAi),
            _ => Err(format!(
                "unsupported provider '{value}'; use openai, openrouter, deepseek, anthropic, ollama, or local-openai"
            )),
        }
    }
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
            Self::DeepSeek => "deepseek",
            Self::Anthropic => "anthropic",
            Self::Ollama => "ollama",
            Self::LocalOpenAi => "local-openai",
        }
    }
    #[must_use]
    pub fn endpoint(self) -> String {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/chat/completions".to_owned(),
            Self::OpenRouter => "https://openrouter.ai/api/v1/chat/completions".to_owned(),
            Self::DeepSeek => "https://api.deepseek.com/chat/completions".to_owned(),
            Self::Anthropic => "https://api.anthropic.com/v1/messages".to_owned(),
            Self::Ollama => format!("{}/v1/chat/completions", ollama_base_url()),
            Self::LocalOpenAi => {
                format!("{}/v1/chat/completions", local_openai_base_url())
            }
        }
    }
    #[must_use]
    pub fn models_endpoint(self) -> String {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/models".to_owned(),
            Self::OpenRouter => "https://openrouter.ai/api/v1/models".to_owned(),
            Self::DeepSeek => "https://api.deepseek.com/models".to_owned(),
            Self::Anthropic => "https://api.anthropic.com/v1/models".to_owned(),
            Self::Ollama => format!("{}/v1/models", ollama_base_url()),
            Self::LocalOpenAi => format!("{}/v1/models", local_openai_base_url()),
        }
    }

    #[must_use]
    pub const fn api_key_environment(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::OpenRouter => "OPENROUTER_API_KEY",
            Self::DeepSeek => "DEEPSEEK_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Ollama => "OLLAMA_HOST",
            Self::LocalOpenAi => "LOCAL_OPENAI_API_KEY",
        }
    }
    #[must_use]
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => "gpt-4.1-mini",
            Self::OpenRouter => "openrouter/free",
            Self::DeepSeek => "deepseek-v4-flash",
            Self::Anthropic => "claude-sonnet-4-5",
            Self::Ollama => "qwen3:8b",
            Self::LocalOpenAi => "local-model",
        }
    }

    #[must_use]
    pub const fn requires_api_key(self) -> bool {
        !matches!(self, Self::Ollama | Self::LocalOpenAi)
    }

    /// All supported providers, in default preference order.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::OpenRouter,
            Self::OpenAi,
            Self::Anthropic,
            Self::DeepSeek,
            Self::Ollama,
            Self::LocalOpenAi,
        ]
    }

    /// Whether a credential for this provider is available from the
    /// environment, the workspace `.env`, or the user credential store.
    #[must_use]
    pub fn has_credential(self) -> bool {
        if self == Self::Ollama {
            return ollama_is_available();
        }
        if self == Self::LocalOpenAi {
            return local_openai_is_available();
        }
        let environment = self.api_key_environment();
        std::env::var(environment).is_ok_and(|value| !value.is_empty())
            || dotenv_value(environment).is_some()
    }
}

fn normalize_base_url(value: &str) -> String {
    let value = value.trim().trim_end_matches('/');
    // Endpoints append `/v1/...`, and OpenAI-compatible servers (LM Studio,
    // vLLM, llama.cpp) commonly display their base as `.../v1`; accept
    // either spelling instead of producing `/v1/v1/chat/completions`.
    let value = value.strip_suffix("/v1").unwrap_or(value).trim_end_matches('/');
    if value.starts_with("http://") || value.starts_with("https://") {
        value.to_owned()
    } else {
        format!("http://{value}")
    }
}

fn ollama_base_url() -> String {
    let host = std::env::var("OLLAMA_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_owned());
    normalize_base_url(&host)
}

fn ollama_is_available() -> bool {
    Client::builder()
        .connect_timeout(Duration::from_millis(250))
        .timeout(Duration::from_millis(500))
        .build()
        .ok()
        .and_then(|client| {
            client
                .get(format!("{}/api/version", ollama_base_url()))
                .send()
                .ok()
        })
        .is_some_and(|response| response.status().is_success())
}

fn local_openai_base_url() -> String {
    std::env::var("LOCAL_OPENAI_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map_or_else(String::new, |value| normalize_base_url(&value))
}

fn local_openai_is_available() -> bool {
    let base_url = local_openai_base_url();
    if base_url.is_empty() {
        return false;
    }
    let Ok(client) = Client::builder()
        .connect_timeout(Duration::from_millis(250))
        .timeout(Duration::from_millis(500))
        .build()
    else {
        return false;
    };
    let mut request = client.get(format!("{base_url}/v1/models"));
    if let Ok(key) = std::env::var("LOCAL_OPENAI_API_KEY")
        && !key.is_empty()
    {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    request
        .send()
        .is_ok_and(|response| response.status().is_success())
}

/// Providers currently usable through an API key or a reachable local runtime.
#[must_use]
pub fn available_providers() -> Vec<ProviderKind> {
    ProviderKind::all()
        .into_iter()
        .filter(|kind| kind.has_credential())
        .collect()
}

/// Authenticated REST provider. The historical type name is retained to keep
/// the public API stable; Anthropic requests branch into their own adapter at
/// the provider edge.
pub struct OpenAiCompatibleProvider {
    kind: ProviderKind,
    api_key: String,
    model: String,
    client: Client,
}

pub struct ProviderRegistry {
    providers: BTreeMap<&'static str, OpenAiCompatibleProvider>,
}

impl ProviderRegistry {
    #[must_use]
    pub fn from_available() -> Self {
        let providers = available_providers()
            .into_iter()
            .filter_map(|kind| {
                OpenAiCompatibleProvider::from_environment(kind, None)
                    .ok()
                    .map(|provider| (kind.name(), provider))
            })
            .collect();
        Self { providers }
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&OpenAiCompatibleProvider> {
        self.providers.get(name)
    }
}

impl OpenAiCompatibleProvider {
    /// # Errors
    ///
    /// Returns an error when the provider credential cannot be found or the
    /// HTTP client cannot be constructed.
    pub fn from_environment(kind: ProviderKind, model: Option<String>) -> Result<Self, String> {
        let api_key = if kind == ProviderKind::Ollama {
            if !ollama_is_available() {
                return Err(format!(
                    "Ollama is not reachable at {}; start Ollama or set OLLAMA_HOST",
                    ollama_base_url()
                ));
            }
            "ollama".to_owned()
        } else if kind == ProviderKind::LocalOpenAi {
            if !local_openai_is_available() {
                return Err(
                    "local OpenAI-compatible server is unavailable; set LOCAL_OPENAI_BASE_URL"
                        .to_owned(),
                );
            }
            std::env::var("LOCAL_OPENAI_API_KEY").unwrap_or_else(|_| "local".to_owned())
        } else {
            let environment = kind.api_key_environment();
            let key = std::env::var(environment)
                .ok()
                .or_else(|| dotenv_value(environment))
                .ok_or_else(|| format!("{environment} is required for provider {}", kind.name()))?;
            if key.is_empty() {
                return Err(format!("{environment} is empty"));
            }
            key
        };
        let client = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(TURN_TIMEOUT)
            .build()
            .map_err(|error| error.to_string())?;
        let requested_model = model.is_some();
        let mut provider = Self {
            kind,
            api_key,
            model: model.unwrap_or_else(|| kind.default_model().to_owned()),
            client,
        };
        // A local runtime may contain any model names. Prefer Junebug's
        // coding default when installed; otherwise make `--provider ollama`
        // immediately useful by selecting the first installed model.
        if matches!(kind, ProviderKind::Ollama | ProviderKind::LocalOpenAi)
            && !requested_model
            && let Ok(models) = provider.list_models()
            && !models.is_empty()
            && !models.contains(&provider.model)
        {
            provider.model.clone_from(&models[0]);
        }
        Ok(provider)
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    /// List model identifiers from the provider's standard `/models`
    /// endpoint, sorted.
    ///
    /// # Errors
    ///
    /// Returns an error for transport failures or an unexpected response
    /// shape.
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let mut request = self
            .client
            .get(self.kind.models_endpoint())
            .timeout(CONNECT_TIMEOUT);
        request = if self.kind == ProviderKind::Anthropic {
            request
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
        } else {
            request.header(AUTHORIZATION, format!("Bearer {}", self.api_key))
        };
        let response = request.send().map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("{} returned {status}", self.kind.name()));
        }
        let body: Value = response.json().map_err(|error| error.to_string())?;
        let mut models = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or("models response lacks a data array")?
            .iter()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        models.sort_unstable();
        Ok(models)
    }
}

/// Read one simple `KEY=value` entry without executing shell syntax,
/// checking the workspace `.env` first and then the user credentials file.
fn dotenv_value(key: &str) -> Option<String> {
    env_file_value(std::path::Path::new(".env"), key)
        .or_else(|| credentials_path().and_then(|path| env_file_value(&path, key)))
        .or_else(|| legacy_credentials_path().and_then(|path| env_file_value(&path, key)))
}

fn env_file_value(path: &std::path::Path, key: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    contents.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        let (candidate, value) = line.split_once('=')?;
        if candidate.trim() != key {
            return None;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .unwrap_or(value);
        let value = value
            .strip_prefix('\'')
            .and_then(|value| value.strip_suffix('\''))
            .unwrap_or(value);
        (!value.is_empty()).then(|| value.to_owned())
    })
}

/// The user-level credential store written by `junebug set`.
#[must_use]
pub fn credentials_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(
        std::path::PathBuf::from(home)
            .join(".junebug")
            .join("credentials.env"),
    )
}

fn legacy_credentials_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(
        std::path::PathBuf::from(home)
            .join(".febo")
            .join("credentials.env"),
    )
}

/// Save `key` for `kind` in the user credential store, replacing any
/// existing entry. Returns the file written.
///
/// # Errors
///
/// Returns an error when the home directory is unknown or the file cannot
/// be written.
pub fn store_credential(kind: ProviderKind, key: &str) -> Result<std::path::PathBuf, String> {
    if !kind.requires_api_key() {
        return Err("local providers are configured through their runtime environment".to_owned());
    }
    let path = credentials_path().ok_or("cannot locate a home directory")?;
    store_credential_at(&path, kind.api_key_environment(), key)?;
    Ok(path)
}

fn store_credential_at(path: &std::path::Path, environment: &str, key: &str) -> Result<(), String> {
    if key.trim().is_empty() {
        return Err("the API key is empty".to_owned());
    }
    if key.contains(['\n', '\r']) {
        return Err("the API key must be a single line".to_owned());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = existing
        .lines()
        .filter(|line| {
            line.split_once('=')
                .is_none_or(|(candidate, _)| candidate.trim() != environment)
        })
        .map(str::to_owned)
        .collect();
    lines.push(format!("{environment}={}", key.trim()));
    let contents = format!("{}\n", lines.join("\n"));
    std::fs::write(path, contents).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, permissions).map_err(|error| error.to_string())?;
    }
    Ok(())
}

impl ModelProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn stream_turn(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<ModelTurn, String> {
        if self.kind == ProviderKind::Anthropic {
            return self.stream_anthropic(model, messages, tools, cancel);
        }
        let request_messages = openai_request_messages(self.kind, messages);
        let mut body = json!({"model": model, "stream": true, "stream_options": {"include_usage": true}, "messages": request_messages});
        // Ollama enables Qwen3 thinking by default. Coding-agent turns need
        // responsive tool calls more than a long hidden reasoning trace; the
        // OpenAI-compatible endpoint documents `none` for this purpose.
        if self.kind == ProviderKind::Ollama {
            body["reasoning_effort"] = Value::String("none".to_owned());
        }
        // OpenAI-compatible endpoints reject an empty tools array, so only
        // send the fields when at least one tool is offered.
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = Value::String("auto".to_owned());
        }
        let mut request = self
            .client
            .post(self.kind.endpoint())
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json");
        if self.kind == ProviderKind::OpenRouter {
            request = request
                .header("HTTP-Referer", "https://github.com/weeksdev/junebug-cli")
                .header("X-OpenRouter-Title", "Junebug CLI");
        }
        let response = request
            .json(&body)
            .send()
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "{} returned {}: {}",
                self.kind.name(),
                status,
                response.text().unwrap_or_default()
            ));
        }
        parse_sse(BufReader::new(response), cancel)
    }
}

impl OpenAiCompatibleProvider {
    fn stream_anthropic(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<ModelTurn, String> {
        let (system, messages) = anthropic_messages(messages);
        let mut body = json!({
            "model": model,
            "max_tokens": 8192,
            "stream": true,
            "messages": messages,
        });
        if !system.is_empty() {
            body["system"] = Value::String(system);
        }
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.iter().filter_map(anthropic_tool).collect());
        }
        let response = self
            .client
            .post(self.kind.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header(CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "anthropic returned {status}: {}",
                response.text().unwrap_or_default()
            ));
        }
        parse_anthropic_sse(BufReader::new(response), cancel)
    }
}

fn anthropic_tool(tool: &Value) -> Option<Value> {
    let function = tool.get("function")?;
    Some(json!({
        "name": function.get("name")?,
        "description": function.get("description").cloned().unwrap_or(Value::String(String::new())),
        "input_schema": function.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"})),
    }))
}

/// Translate Junebug's canonical OpenAI-shaped transcript at the provider edge.
/// Keeping this conversion here allows a conversation to switch providers
/// between turns without changing the session or agent-loop formats.
fn anthropic_messages(messages: &[Value]) -> (String, Vec<Value>) {
    let mut system = Vec::new();
    let mut translated = Vec::<Value>::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => {
                if let Some(text) = message.get("content").and_then(Value::as_str) {
                    system.push(text.to_owned());
                }
            }
            "user" => translated.push(json!({
                "role": "user",
                "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
            })),
            "assistant" => {
                let mut content = Vec::new();
                if let Some(text) = message
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    content.push(json!({"type": "text", "text": text}));
                }
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let Some(function) = call.get("function") else {
                            continue;
                        };
                        let arguments = function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .and_then(|value| serde_json::from_str(value).ok())
                            .unwrap_or_else(|| json!({}));
                        content.push(json!({
                            "type": "tool_use",
                            "id": call.get("id").cloned().unwrap_or(Value::String(String::new())),
                            "name": function.get("name").cloned().unwrap_or(Value::String(String::new())),
                            "input": arguments,
                        }));
                    }
                }
                if !content.is_empty() {
                    translated.push(json!({"role": "assistant", "content": content}));
                }
            }
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                    "is_error": message.get("content").and_then(Value::as_str).is_some_and(|text| text.starts_with("ERROR:")),
                });
                if let Some(last) = translated.last_mut()
                    && last.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
                    && content
                        .first()
                        .and_then(|item| item.get("type"))
                        .and_then(Value::as_str)
                        == Some("tool_result")
                {
                    content.push(block);
                } else {
                    translated.push(json!({"role": "user", "content": [block]}));
                }
            }
            _ => {}
        }
    }
    (system.join("\n\n"), translated)
}

fn has_assistant_payload(message: &Value) -> bool {
    let has_content = match message.get("content") {
        Some(Value::String(content)) => !content.is_empty(),
        Some(Value::Array(content)) => !content.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    };
    let has_tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());
    has_content || has_tool_calls
}

/// Repair invalid empty assistant entries left by interrupted/empty legacy
/// streams. Reasoning is retained only for `DeepSeek`, whose thinking-mode
/// tool continuation requires it; other OpenAI-shaped providers may reject
/// that provider-specific field.
fn openai_request_messages(kind: ProviderKind, messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .filter_map(|message| {
            let mut message = message.clone();
            if message.get("role").and_then(Value::as_str) == Some("assistant") {
                if !has_assistant_payload(&message) {
                    return None;
                }
                if kind != ProviderKind::DeepSeek
                    && let Some(object) = message.as_object_mut()
                {
                    object.remove("reasoning_content");
                }
            }
            Some(message)
        })
        .collect()
}

#[derive(Default)]
struct AnthropicToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[allow(clippy::too_many_lines)]
fn parse_anthropic_sse(reader: impl BufRead, cancel: &AtomicBool) -> Result<ModelTurn, String> {
    let mut text_deltas = Vec::new();
    let mut text = String::new();
    let mut partial_calls = BTreeMap::<usize, AnthropicToolCall>::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut interrupted = false;
    for line in reader.lines() {
        if cancel.load(Ordering::Relaxed) {
            partial_calls.clear();
            interrupted = true;
            break;
        }
        let line = line.map_err(|error| error.to_string())?;
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        let event: Value = serde_json::from_str(payload)
            .map_err(|error| format!("invalid Anthropic SSE JSON: {error}"))?;
        if event.get("type").and_then(Value::as_str) == Some("error") {
            return Err(format!("provider stream error: {}", event["error"]));
        }
        input_tokens = event
            .pointer("/message/usage/input_tokens")
            .or_else(|| event.pointer("/usage/input_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(input_tokens);
        output_tokens = event
            .pointer("/message/usage/output_tokens")
            .or_else(|| event.pointer("/usage/output_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(output_tokens);
        let index = event
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        match event.get("type").and_then(Value::as_str) {
            Some("content_block_start")
                if event.pointer("/content_block/type").and_then(Value::as_str)
                    == Some("tool_use") =>
            {
                partial_calls.insert(
                    index,
                    AnthropicToolCall {
                        id: event
                            .pointer("/content_block/id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        name: event
                            .pointer("/content_block/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        arguments: String::new(),
                    },
                );
            }
            Some("content_block_delta") => {
                if let Some(delta) = event.pointer("/delta/text").and_then(Value::as_str) {
                    text.push_str(delta);
                    text_deltas.push(delta.to_owned());
                }
                if let Some(delta) = event.pointer("/delta/partial_json").and_then(Value::as_str)
                    && let Some(call) = partial_calls.get_mut(&index)
                {
                    call.arguments.push_str(delta);
                }
            }
            _ => {}
        }
    }
    if interrupted {
        text.push_str("\n[response interrupted by user]");
    }
    let tool_calls = partial_calls
        .into_values()
        .filter(|call| !call.name.is_empty())
        .map(|call| ToolCall {
            id: call.id,
            name: call.name,
            arguments: if call.arguments.is_empty() {
                "{}".to_owned()
            } else {
                call.arguments
            },
        })
        .collect::<Vec<_>>();
    let mut assistant_message = json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text) },
    });
    if !tool_calls.is_empty() {
        assistant_message["tool_calls"] = Value::Array(
            tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    })
                })
                .collect(),
        );
    }
    Ok(ModelTurn {
        text_deltas,
        tool_calls,
        assistant_message,
        input_tokens,
        output_tokens,
    })
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[allow(clippy::too_many_lines)]
fn parse_sse(reader: impl BufRead, cancel: &AtomicBool) -> Result<ModelTurn, String> {
    let mut text_deltas = Vec::new();
    let mut text = String::new();
    let mut reasoning_content = String::new();
    let mut partial_calls = BTreeMap::<usize, PartialToolCall>::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut interrupted = false;
    for line in reader.lines() {
        if cancel.load(Ordering::Relaxed) {
            // Drop partial tool calls: their JSON arguments may be
            // incomplete and they must never execute after an interrupt.
            partial_calls.clear();
            interrupted = true;
            break;
        }
        let line = line.map_err(|error| error.to_string())?;
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let chunk: Value =
            serde_json::from_str(payload).map_err(|error| format!("invalid SSE JSON: {error}"))?;
        if let Some(error) = chunk.get("error") {
            return Err(format!("provider stream error: {error}"));
        }
        if let Some(content) = chunk
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            text.push_str(content);
            text_deltas.push(content.to_owned());
        }
        if let Some(reasoning) = chunk
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            reasoning_content.push_str(reasoning);
        }
        if let Some(calls) = chunk
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for call in calls {
                let index = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0);
                let partial = partial_calls.entry(index).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    id.clone_into(&mut partial.id);
                }
                if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                    partial.name.push_str(name);
                }
                if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str)
                {
                    partial.arguments.push_str(arguments);
                }
            }
        }
        if let Some(usage) = chunk.get("usage") {
            input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(input_tokens);
            output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(output_tokens);
        }
    }
    let tool_calls = partial_calls
        .into_values()
        .filter(|call| !call.name.is_empty())
        .map(|call| ToolCall {
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        })
        .collect::<Vec<_>>();
    if interrupted {
        // Leave a marker in the recorded history (not in the streamed
        // deltas) so the model knows the reply was cut short.
        text.push_str("\n[response interrupted by user]");
    }
    let mut assistant_message = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { Value::String(text) }});
    if !reasoning_content.is_empty() {
        assistant_message["reasoning_content"] = Value::String(reasoning_content);
    }
    if !tool_calls.is_empty() {
        assistant_message["tool_calls"] = Value::Array(tool_calls.iter().map(|call| json!({"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}})).collect());
    }
    Ok(ModelTurn {
        text_deltas,
        tool_calls,
        assistant_message,
        input_tokens,
        output_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderKind, anthropic_messages, env_file_value, normalize_base_url,
        openai_request_messages, parse_anthropic_sse, parse_sse, store_credential_at,
    };
    use serde_json::json;
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_known_providers() {
        assert_eq!(
            ProviderKind::parse("openrouter"),
            Ok(ProviderKind::OpenRouter)
        );
        assert_eq!(ProviderKind::parse("claude"), Ok(ProviderKind::Anthropic));
        assert_eq!(ProviderKind::parse("local"), Ok(ProviderKind::Ollama));
        assert_eq!(
            ProviderKind::parse("lmstudio"),
            Ok(ProviderKind::LocalOpenAi)
        );
        assert!(!ProviderKind::Ollama.requires_api_key());
        assert!(!ProviderKind::LocalOpenAi.requires_api_key());
        assert_eq!(ProviderKind::Ollama.default_model(), "qwen3:8b");
        assert!(ProviderKind::parse("fake").is_err());
    }

    #[test]
    fn base_url_normalization_accepts_v1_and_slash_variants() {
        // Local OpenAI-compatible servers advertise their endpoint with and
        // without `/v1`; every spelling must yield the same origin so the
        // appended `/v1/chat/completions` never doubles up.
        for raw in [
            "localhost:1234",
            "http://localhost:1234",
            "http://localhost:1234/",
            "http://localhost:1234/v1",
            "http://localhost:1234/v1/",
            " http://localhost:1234/v1 ",
        ] {
            assert_eq!(normalize_base_url(raw), "http://localhost:1234", "{raw:?}");
        }
        // A nested API base keeps its prefix: re-appending `/v1` restores it.
        assert_eq!(
            normalize_base_url("https://gateway.example/api/v1"),
            "https://gateway.example/api"
        );
        // `/v1` only strips as a whole path segment.
        assert_eq!(
            normalize_base_url("http://host/model-v1"),
            "http://host/model-v1"
        );
    }

    #[test]
    fn translates_canonical_tool_history_for_anthropic() {
        let messages = vec![
            json!({"role":"system","content":"be careful"}),
            json!({"role":"user","content":"read it"}),
            json!({"role":"assistant","content":null,"tool_calls":[
                {"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}},
                {"id":"call-2","type":"function","function":{"name":"git_status","arguments":"{}"}}
            ]}),
            json!({"role":"tool","tool_call_id":"call-1","content":"contents"}),
            json!({"role":"tool","tool_call_id":"call-2","content":"ERROR: unavailable"}),
        ];
        let (system, translated) = anthropic_messages(&messages);
        assert_eq!(system, "be careful");
        assert_eq!(translated[1]["content"][0]["type"], "tool_use");
        assert_eq!(translated[1]["content"][0]["input"]["path"], "README.md");
        assert_eq!(translated[2]["role"], "user");
        assert_eq!(translated[2]["content"].as_array().map(Vec::len), Some(2));
        assert_eq!(translated[2]["content"][1]["tool_use_id"], "call-2");
        assert_eq!(translated[2]["content"][1]["is_error"], true);
    }

    #[test]
    fn parses_anthropic_text_tools_and_usage_into_canonical_turn() {
        let stream = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read_file\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":9}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        );
        let turn = parse_anthropic_sse(Cursor::new(stream), &AtomicBool::new(false))
            .expect("valid Anthropic stream");
        assert_eq!(turn.text_deltas, ["Checking"]);
        assert_eq!(turn.input_tokens, 12);
        assert_eq!(turn.output_tokens, 9);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"path":"README.md"}"#);
        assert_eq!(turn.assistant_message["tool_calls"][0]["id"], "toolu_1");
    }

    #[test]
    fn repairs_empty_assistant_history_and_scopes_deepseek_reasoning() {
        let messages = vec![
            json!({"role":"system","content":"system"}),
            json!({"role":"assistant","content":null}),
            json!({"role":"assistant","content":""}),
            json!({"role":"assistant","content":null,"reasoning_content":"thinking","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"user","content":"continue"}),
        ];
        let deepseek = openai_request_messages(ProviderKind::DeepSeek, &messages);
        assert_eq!(deepseek.len(), 3);
        assert_eq!(deepseek[1]["reasoning_content"], "thinking");
        let openai = openai_request_messages(ProviderKind::OpenAi, &messages);
        assert_eq!(openai.len(), 3);
        assert!(openai[1].get("reasoning_content").is_none());
    }

    #[test]
    fn openai_stream_preserves_reasoning_for_tool_continuation() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"inspect first\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let turn = parse_sse(Cursor::new(stream), &AtomicBool::new(false)).expect("valid stream");
        assert_eq!(turn.assistant_message["content"], serde_json::Value::Null);
        assert_eq!(turn.assistant_message["reasoning_content"], "inspect first");
        assert_eq!(turn.tool_calls.len(), 1);
    }

    #[test]
    fn stores_and_replaces_credentials() {
        let path = std::env::temp_dir()
            .join(format!(
                "junebug-credentials-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ))
            .join("credentials.env");
        store_credential_at(&path, "DEEPSEEK_API_KEY", "first").expect("store");
        store_credential_at(&path, "OPENAI_API_KEY", "other").expect("store");
        store_credential_at(&path, "DEEPSEEK_API_KEY", "second").expect("replace");
        assert_eq!(
            env_file_value(&path, "DEEPSEEK_API_KEY").as_deref(),
            Some("second")
        );
        assert_eq!(
            env_file_value(&path, "OPENAI_API_KEY").as_deref(),
            Some("other")
        );
        assert!(store_credential_at(&path, "X", "multi\nline").is_err());
        std::fs::remove_dir_all(path.parent().expect("parent")).expect("cleanup");
    }
}
