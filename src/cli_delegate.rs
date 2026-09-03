//! `claude-cli` / `codex-cli` providers: instead of an HTTP call, a turn is
//! delegated whole to a local, already-authenticated `claude` or `codex`
//! binary running non-interactively, and its final answer is returned as if
//! it were a normal (tool-call-free) assistant reply.
//!
//! This exists to use a Claude Pro/Max or `ChatGPT` Plus/Pro/Codex
//! subscription from Junebug without reimplementing anyone's OAuth. Both
//! vendors' terms reserve subscription-authenticated access for their own
//! official clients — Anthropic's Agent SDK docs say so explicitly: "Unless
//! previously approved, Anthropic does not allow third party developers to
//! offer claude.ai login or rate limits for their products." Driving the
//! CLI you personally already ran `claude login`/`codex login` on is a
//! different thing: no credential is brokered or extracted, Junebug just
//! shells out the same way it already does for `rg` and `ollama`. It is
//! also Anthropic's own documented integration path for non-Python/TS
//! languages ("For other languages, run the CLI programmatically with the
//! `-p` flag and `--output-format json`" — Agent SDK overview).
//!
//! Because the delegate runs its own built-in agent loop (its own Read /
//! Write / Bash tools, its own approval system), it does not participate in
//! Junebug's tool loop at all: `stream_turn` always returns zero
//! `tool_calls`, and the `tools` argument is ignored. Junebug's
//! `PolicyEngine` is not consulted for what the delegate does internally —
//! there is no per-tool-call channel to intercept, since the whole turn is
//! one non-interactive subprocess invocation. Junebug's permission mode is
//! instead mapped once per turn onto each CLI's own sandbox/approval flags
//! (see `effective_sandbox`), and the pre-prompt workspace checkpoint that
//! already runs before every turn remains the safety net for `/rewind`.
//!
//! Multi-turn continuity uses each CLI's own session/thread resume
//! mechanism (`--resume` / `codex exec resume`) rather than replaying
//! Junebug's full message history as prompt text: only the latest user
//! message is sent each turn, and the delegate's own server-side history
//! (and prompt caching) carries the rest. Switching the sandbox mode
//! (changing Junebug's permission) starts a fresh delegate session rather
//! than resuming, since neither CLI allows changing sandbox/approval flags
//! on a resumed session.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::PermissionMode;
use crate::provider::{ModelProvider, ModelTurn, ProviderKind};

/// Upper bound on one delegate turn: the CLI runs its own agentic loop with
/// no way for Junebug to see progress, so a hang must still end eventually
/// even without a user Esc. Generous because a large refactor genuinely
/// takes a while.
const DELEGATE_TIMEOUT: Duration = Duration::from_mins(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DelegateSandbox {
    ReadOnly,
    WorkspaceWrite,
    Full,
}

/// Junebug's live permission mode (plus the static plan-mode guard) mapped
/// onto a coarse sandbox tier. Neither CLI can be interactively asked for
/// per-action approval headlessly, so `Ask` collapses to `ReadOnly` — the
/// safe side — rather than silently acting without a real human decision.
fn effective_sandbox(permission: PermissionMode, plan: bool) -> DelegateSandbox {
    if plan {
        return DelegateSandbox::ReadOnly;
    }
    match permission {
        PermissionMode::ReadOnly | PermissionMode::Ask => DelegateSandbox::ReadOnly,
        PermissionMode::WorkspaceWrite => DelegateSandbox::WorkspaceWrite,
        PermissionMode::Yolo => DelegateSandbox::Full,
    }
}

struct DelegateSession {
    id: String,
    sandbox: DelegateSandbox,
}

pub struct CliDelegateProvider {
    kind: ProviderKind,
    workspace: PathBuf,
    /// `None` means "let the CLI pick its own default model."
    model_override: Option<String>,
    plan: bool,
    permission: Mutex<PermissionMode>,
    session: Mutex<Option<DelegateSession>>,
}

impl CliDelegateProvider {
    /// # Errors
    ///
    /// Returns an error when `kind` is not a delegate kind.
    pub fn new(
        kind: ProviderKind,
        workspace: PathBuf,
        model: Option<String>,
        permission: PermissionMode,
        plan: bool,
    ) -> Result<Self, String> {
        if !kind.is_cli_delegate() {
            return Err(format!("{} is not a CLI delegate provider", kind.name()));
        }
        Ok(Self {
            kind,
            workspace,
            model_override: normalize_model(model),
            plan,
            permission: Mutex::new(permission),
            session: Mutex::new(None),
        })
    }

    #[must_use]
    pub fn model(&self) -> &str {
        self.model_override.as_deref().unwrap_or("default")
    }

    /// Changing the model starts a fresh delegate session next turn: the
    /// old session's history was built on the previous model.
    pub fn set_model(&mut self, model: String) {
        self.model_override = normalize_model(Some(model));
        self.session = Mutex::new(None);
    }

    pub fn set_permission(&self, permission: PermissionMode) {
        if let Ok(mut guard) = self.permission.lock() {
            *guard = permission;
        }
    }

    /// Always errors: there is no live model catalog to fetch for a local
    /// CLI delegate. Callers already treat a `list_models` error as
    /// best-effort (see `provider::OpenAiCompatibleProvider` callers in
    /// `main.rs`), so this falls through to a direct model-name switch or a
    /// single "default" picker entry rather than a live list.
    ///
    /// # Errors
    ///
    /// Always returns an error, by design.
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        Err(format!(
            "{} has no live model catalog — pass a model name directly (e.g. /model {}:opus) \
             or leave it as \"default\" to let the CLI choose",
            self.kind.name(),
            self.kind.name()
        ))
    }
}

fn normalize_model(model: Option<String>) -> Option<String> {
    model.filter(|value| !value.is_empty() && value != "default")
}

impl ModelProvider for CliDelegateProvider {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn stream_turn(
        &self,
        _model: &str,
        messages: &[Value],
        _tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<ModelTurn, String> {
        use std::fmt::Write as _;
        let prompt = latest_user_text(messages)
            .ok_or_else(|| "no user message to send to the delegate CLI".to_owned())?;
        let permission = *self
            .permission
            .lock()
            .map_err(|_| "permission lock poisoned".to_owned())?;
        let sandbox = effective_sandbox(permission, self.plan);

        let mut session_guard = self
            .session
            .lock()
            .map_err(|_| "session lock poisoned".to_owned())?;
        let resume_id = session_guard
            .as_ref()
            .filter(|session| session.sandbox == sandbox)
            .map(|session| session.id.clone());

        let model = self.model_override.as_deref().unwrap_or("");
        let command = match self.kind {
            ProviderKind::ClaudeCli => build_claude_command(
                &self.workspace,
                model,
                sandbox,
                resume_id.as_deref(),
                &prompt,
            ),
            ProviderKind::CodexCli => build_codex_command(
                &self.workspace,
                model,
                sandbox,
                resume_id.as_deref(),
                &prompt,
            ),
            other => {
                return Err(format!("{} is not a CLI delegate provider", other.name()));
            }
        };

        let (raw_output, stop, success) = run_and_capture(self.kind, command, cancel)?;
        let parsed = match self.kind {
            ProviderKind::ClaudeCli => parse_claude_output(&raw_output),
            ProviderKind::CodexCli => parse_codex_output(&raw_output),
            _ => unreachable!("checked above"),
        };

        if let Some(id) = parsed.session_id.clone() {
            *session_guard = Some(DelegateSession { id, sandbox });
        }
        drop(session_guard);

        if matches!(stop, StopReason::Completed) && (!success || parsed.is_error) {
            return Err(if parsed.text.trim().is_empty() {
                let tail = tail_chars(raw_output.trim(), 2000);
                if tail.is_empty() {
                    format!(
                        "{} exited with an error and produced no output",
                        self.kind.name()
                    )
                } else {
                    format!("{} exited with an error:\n{tail}", self.kind.name())
                }
            } else {
                parsed.text
            });
        }

        let mut text = parsed.text;
        match stop {
            StopReason::Cancelled => text.push_str("\n[response interrupted by user]"),
            StopReason::TimedOut => {
                let _ = write!(
                    text,
                    "\n[{} timed out after {}s and was killed]",
                    self.kind.name(),
                    DELEGATE_TIMEOUT.as_secs()
                );
            }
            StopReason::Completed => {}
        }

        let assistant_message = json!({
            "role": "assistant",
            "content": if text.is_empty() { Value::Null } else { Value::String(text.clone()) },
        });
        Ok(ModelTurn {
            text_deltas: vec![text],
            tool_calls: Vec::new(),
            assistant_message,
            input_tokens: parsed.input_tokens,
            output_tokens: parsed.output_tokens,
        })
    }
}

/// The most recent user message's text content, joined if it was structured
/// as content blocks.
fn latest_user_text(messages: &[Value]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            return None;
        }
        match message.get("content") {
            Some(Value::String(text)) => Some(text.clone()),
            Some(Value::Array(blocks)) => {
                let joined = blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                (!joined.is_empty()).then_some(joined)
            }
            _ => None,
        }
    })
}

fn build_claude_command(
    workspace: &std::path::Path,
    model: &str,
    sandbox: DelegateSandbox,
    resume: Option<&str>,
    prompt: &str,
) -> Command {
    let mut command = Command::new("claude");
    command.current_dir(workspace);
    command.arg("-p");
    // Disables CLAUDE.md/skills/plugins/hooks/custom-commands and (load-
    // bearing for this feature) auto-memory — a bare non-safe-mode `-p` run
    // was observed live writing a real memory file from a throwaway test
    // prompt. Auth (subscription OAuth or API key) still works normally
    // under --safe-mode, unlike --bare, which forces API-key-only auth.
    command.arg("--safe-mode");
    command
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose");
    if !model.is_empty() {
        command.arg("--model").arg(model);
    }
    command.arg("--permission-mode").arg(match sandbox {
        DelegateSandbox::ReadOnly => "plan",
        DelegateSandbox::WorkspaceWrite => "acceptEdits",
        DelegateSandbox::Full => "bypassPermissions",
    });
    if let Some(id) = resume {
        command.arg("--resume").arg(id);
    }
    // The prompt is always the last argument: variadic flags like
    // --allowedTools greedily consume following tokens (verified live —
    // `--allowedTools "Bash(ls:*)" "<prompt>"` swallowed the prompt itself
    // as another tool name), so none of those are used here.
    command.arg(prompt);
    command
}

fn build_codex_command(
    workspace: &std::path::Path,
    model: &str,
    sandbox: DelegateSandbox,
    resume: Option<&str>,
    prompt: &str,
) -> Command {
    let mut command = Command::new("codex");
    command.current_dir(workspace);
    command.arg("exec");
    if let Some(id) = resume {
        // `codex exec resume` does not accept -C/--sandbox: both are fixed
        // from the session it started with.
        command.arg("resume").arg(id);
    } else {
        command.arg("-C").arg(workspace);
        command.arg("--sandbox").arg(match sandbox {
            DelegateSandbox::ReadOnly => "read-only",
            DelegateSandbox::WorkspaceWrite => "workspace-write",
            DelegateSandbox::Full => "danger-full-access",
        });
    }
    command.arg("--json").arg("--skip-git-repo-check");
    if sandbox == DelegateSandbox::Full {
        command.arg("--dangerously-bypass-approvals-and-sandbox");
    }
    if !model.is_empty() {
        command.arg("--model").arg(model);
    }
    command.arg(prompt);
    command
}

enum StopReason {
    Completed,
    Cancelled,
    TimedOut,
}

/// Spawn `command`, polling for exit while watching `cancel` and an overall
/// timeout, then return whatever combined stdout+stderr was captured.
/// Reuses `tool::run_command_with_access`'s subprocess primitives (bounded,
/// non-hanging output capture; whole-process-tree kill) so every place
/// Junebug shells out behaves the same way under interruption.
fn run_and_capture(
    kind: ProviderKind,
    mut command: Command,
    cancel: &AtomicBool,
) -> Result<(String, StopReason, bool), String> {
    use std::process::Stdio;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = command
        .spawn()
        .map_err(|error| describe_spawn_error(kind, &error))?;
    let stdout = crate::tool::drain_stream(child.stdout.take());
    let stderr = crate::tool::drain_stream(child.stderr.take());
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Err(error) => return Err(error.to_string()),
            Ok(Some(status)) => {
                let (text, _) = crate::tool::collect_output(&stdout, &stderr);
                return Ok((text, StopReason::Completed, status.success()));
            }
            Ok(None) => {}
        }
        if cancel.load(Ordering::Relaxed) {
            crate::tool::kill_command_tree(&mut child);
            let _ = child.wait();
            let (text, _) = crate::tool::collect_output(&stdout, &stderr);
            return Ok((text, StopReason::Cancelled, false));
        }
        if started.elapsed() >= DELEGATE_TIMEOUT {
            crate::tool::kill_command_tree(&mut child);
            let _ = child.wait();
            let (text, _) = crate::tool::collect_output(&stdout, &stderr);
            return Ok((text, StopReason::TimedOut, false));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn describe_spawn_error(kind: ProviderKind, error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        format!(
            "{} is not installed or not on PATH; install it and log in ({} login) first",
            kind.name(),
            match kind {
                ProviderKind::ClaudeCli => "claude",
                _ => "codex",
            }
        )
    } else {
        format!("cannot run {}: {error}", kind.name())
    }
}

fn tail_chars(text: &str, cap: usize) -> String {
    let count = text.chars().count();
    text.chars()
        .skip(count.saturating_sub(cap))
        .collect::<String>()
        .trim()
        .to_owned()
}

struct ParsedTurn {
    text: String,
    is_error: bool,
    session_id: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
}

#[allow(clippy::cast_possible_truncation)]
fn token_count(value: &Value, pointer: &str) -> u32 {
    value.pointer(pointer).and_then(Value::as_u64).unwrap_or(0) as u32
}

/// Parse Claude Code's `--output-format stream-json` event lines. The final
/// `"type":"result"` line (identical in shape to plain `--output-format
/// json`) is authoritative; if the process was killed before that line
/// arrived (cancel/timeout), fall back to whatever assistant text blocks
/// streamed by, plus the session id from the earlier `system`/`init` line so
/// even an interrupted turn can still be resumed.
fn parse_claude_output(raw: &str) -> ParsedTurn {
    let mut text_parts = Vec::new();
    let mut fallback_session_id = None;
    for line in raw.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if fallback_session_id.is_none()
            && let Some(id) = value.get("session_id").and_then(Value::as_str)
        {
            fallback_session_id = Some(id.to_owned());
        }
        match value.get("type").and_then(Value::as_str) {
            Some("assistant") => {
                let Some(blocks) = value.pointer("/message/content").and_then(Value::as_array)
                else {
                    continue;
                };
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                text_parts.push(text.to_owned());
                            }
                        }
                        Some("tool_use") => {
                            if let Some(name) = block.get("name").and_then(Value::as_str) {
                                text_parts.push(format!("[used tool: {name}]"));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("result") => {
                let text = value
                    .get("result")
                    .and_then(Value::as_str)
                    .map_or_else(|| text_parts.join("\n\n"), str::to_owned);
                return ParsedTurn {
                    text,
                    is_error: value
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    session_id: value
                        .get("session_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .or(fallback_session_id),
                    input_tokens: token_count(&value, "/usage/input_tokens"),
                    output_tokens: token_count(&value, "/usage/output_tokens"),
                };
            }
            _ => {}
        }
    }
    ParsedTurn {
        text: text_parts.join("\n\n"),
        is_error: false,
        session_id: fallback_session_id,
        input_tokens: 0,
        output_tokens: 0,
    }
}

/// Parse Codex's `exec --json` event lines: `agent_message`/
/// `command_execution` items narrate what happened, `turn.completed` closes
/// out with usage. No `turn.completed` (killed early, or a genuine failure)
/// is treated as an error/incomplete turn by the caller.
fn parse_codex_output(raw: &str) -> ParsedTurn {
    let mut text_parts = Vec::new();
    let mut thread_id = None;
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut completed = false;
    for line in raw.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("thread.started") => {
                thread_id = value
                    .get("thread_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            Some("item.completed") => {
                let Some(item) = value.get("item") else {
                    continue;
                };
                match item.get("type").and_then(Value::as_str) {
                    Some("agent_message") => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            text_parts.push(text.to_owned());
                        }
                    }
                    Some("command_execution") => {
                        if let Some(command) = item.get("command").and_then(Value::as_str) {
                            text_parts.push(format!("[ran: {command}]"));
                        }
                    }
                    _ => {}
                }
            }
            Some("turn.completed") => {
                completed = true;
                input_tokens = token_count(&value, "/usage/input_tokens");
                output_tokens = token_count(&value, "/usage/output_tokens");
            }
            Some("turn.failed" | "error") => {
                if let Some(message) = value
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| value.pointer("/error/message").and_then(Value::as_str))
                {
                    text_parts.push(format!("[error: {message}]"));
                }
            }
            _ => {}
        }
    }
    ParsedTurn {
        text: text_parts.join("\n\n"),
        is_error: !completed,
        session_id: thread_id,
        input_tokens,
        output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_sandbox, parse_claude_output, parse_codex_output};
    use crate::PermissionMode;

    #[test]
    fn plan_mode_always_forces_read_only_regardless_of_permission() {
        assert!(matches!(
            effective_sandbox(PermissionMode::Yolo, true),
            super::DelegateSandbox::ReadOnly
        ));
    }

    #[test]
    fn ask_collapses_to_read_only_since_headless_cannot_prompt() {
        assert!(matches!(
            effective_sandbox(PermissionMode::Ask, false),
            super::DelegateSandbox::ReadOnly
        ));
    }

    #[test]
    fn yolo_maps_to_full_sandbox() {
        assert!(matches!(
            effective_sandbox(PermissionMode::Yolo, false),
            super::DelegateSandbox::Full
        ));
    }

    #[test]
    fn parses_claude_stream_json_result_line() {
        let raw = r#"{"type":"system","subtype":"init","session_id":"abc"}
{"type":"assistant","message":{"content":[{"type":"text","text":"draft"}]},"session_id":"abc"}
{"type":"result","result":"PONG","is_error":false,"session_id":"abc","usage":{"input_tokens":2,"output_tokens":5}}"#;
        let parsed = parse_claude_output(raw);
        assert_eq!(parsed.text, "PONG");
        assert!(!parsed.is_error);
        assert_eq!(parsed.session_id.as_deref(), Some("abc"));
        assert_eq!(parsed.input_tokens, 2);
        assert_eq!(parsed.output_tokens, 5);
    }

    #[test]
    fn falls_back_to_assistant_text_when_no_result_line_arrived() {
        let raw = r#"{"type":"system","subtype":"init","session_id":"abc"}
{"type":"assistant","message":{"content":[{"type":"text","text":"partial answer"}]},"session_id":"abc"}"#;
        let parsed = parse_claude_output(raw);
        assert_eq!(parsed.text, "partial answer");
        assert_eq!(parsed.session_id.as_deref(), Some("abc"));
    }

    #[test]
    fn parses_codex_exec_json_events() {
        let raw = r#"{"type":"thread.started","thread_id":"t1"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hi"}}
{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":4}}"#;
        let parsed = parse_codex_output(raw);
        assert_eq!(parsed.text, "hi");
        assert!(!parsed.is_error);
        assert_eq!(parsed.session_id.as_deref(), Some("t1"));
        assert_eq!(parsed.input_tokens, 3);
        assert_eq!(parsed.output_tokens, 4);
    }

    #[test]
    fn codex_without_turn_completed_is_treated_as_incomplete() {
        let raw = r#"{"type":"thread.started","thread_id":"t1"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"still going"}}"#;
        let parsed = parse_codex_output(raw);
        assert!(parsed.is_error);
        assert_eq!(parsed.session_id.as_deref(), Some("t1"));
    }

    #[test]
    fn skips_unparseable_lines_instead_of_failing() {
        let raw = "not json\n{\"type\":\"turn.completed\",\"usage\":{}}\ntrailing garbage {";
        let parsed = parse_codex_output(raw);
        assert!(!parsed.is_error);
    }
}
