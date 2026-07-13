//! Provider-neutral streaming model contract and OpenAI-compatible REST client.

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
            _ => Err(format!(
                "unsupported provider '{value}'; use openai, openrouter, or deepseek"
            )),
        }
    }
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
            Self::DeepSeek => "deepseek",
        }
    }
    #[must_use]
    pub const fn endpoint(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/chat/completions",
            Self::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
            Self::DeepSeek => "https://api.deepseek.com/chat/completions",
        }
    }
    #[must_use]
    pub const fn models_endpoint(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/models",
            Self::OpenRouter => "https://openrouter.ai/api/v1/models",
            Self::DeepSeek => "https://api.deepseek.com/models",
        }
    }

    #[must_use]
    pub const fn api_key_environment(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::OpenRouter => "OPENROUTER_API_KEY",
            Self::DeepSeek => "DEEPSEEK_API_KEY",
        }
    }
    #[must_use]
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => "gpt-4.1-mini",
            Self::OpenRouter => "openrouter/free",
            Self::DeepSeek => "deepseek-v4-flash",
        }
    }

    /// All supported providers, in default preference order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::OpenRouter, Self::OpenAi, Self::DeepSeek]
    }

    /// Whether a credential for this provider is available from the
    /// environment, the workspace `.env`, or the user credential store.
    #[must_use]
    pub fn has_credential(self) -> bool {
        let environment = self.api_key_environment();
        std::env::var(environment).is_ok_and(|value| !value.is_empty())
            || dotenv_value(environment).is_some()
    }
}

/// Providers that currently have a usable credential, in preference order.
#[must_use]
pub fn available_providers() -> Vec<ProviderKind> {
    ProviderKind::all()
        .into_iter()
        .filter(|kind| kind.has_credential())
        .collect()
}

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
        let environment = kind.api_key_environment();
        let api_key = std::env::var(environment)
            .ok()
            .or_else(|| dotenv_value(environment))
            .ok_or_else(|| format!("{environment} is required for provider {}", kind.name()))?;
        if api_key.is_empty() {
            return Err(format!("{environment} is empty"));
        }
        let client = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(TURN_TIMEOUT)
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            kind,
            api_key,
            model: model.unwrap_or_else(|| kind.default_model().to_owned()),
            client,
        })
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
        let response = self
            .client
            .get(self.kind.models_endpoint())
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .send()
            .map_err(|error| error.to_string())?;
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

/// The user-level credential store written by `febo set`.
#[must_use]
pub fn credentials_path() -> Option<std::path::PathBuf> {
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
        let mut body = json!({"model": model, "stream": true, "stream_options": {"include_usage": true}, "messages": messages});
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
                .header("HTTP-Referer", "https://github.com/weeksdev/febo_cli")
                .header("X-OpenRouter-Title", "Febo CLI");
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
        parse_sse(response, cancel)
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn parse_sse(
    response: reqwest::blocking::Response,
    cancel: &AtomicBool,
) -> Result<ModelTurn, String> {
    let mut text_deltas = Vec::new();
    let mut text = String::new();
    let mut partial_calls = BTreeMap::<usize, PartialToolCall>::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut interrupted = false;
    for line in BufReader::new(response).lines() {
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
    use super::{ProviderKind, env_file_value, store_credential_at};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_known_providers() {
        assert_eq!(
            ProviderKind::parse("openrouter"),
            Ok(ProviderKind::OpenRouter)
        );
        assert!(ProviderKind::parse("fake").is_err());
    }

    #[test]
    fn stores_and_replaces_credentials() {
        let path = std::env::temp_dir()
            .join(format!(
                "febo-credentials-{}",
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
