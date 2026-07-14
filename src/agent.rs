//! The model-driven tool loop and tool gateway. Every tool invocation,
//! including MCP tools, is routed through the `PolicyEngine` before it
//! touches the workspace, a subprocess, or an MCP server.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};

use crate::context;
use crate::mcp;
use crate::policy::{Decision, PolicyEngine};
use crate::provider::{ModelProvider, ToolCall};
use crate::router::RouteDecision;
use crate::session::SessionWriter;
use crate::tool::{BUILTIN_TOOLS, ToolRisk, Workspace};

pub struct McpClient {
    pub name: String,
    pub client: mcp::Client,
}

#[derive(Debug, Clone)]
pub struct LoopOutcome {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub interrupted: bool,
    pub provider: String,
    pub model: String,
    pub band: Option<String>,
    pub switches: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TurnState {
    pub turn_index: usize,
    pub turns_remaining: usize,
    pub consecutive_tool_failures: usize,
}

pub struct Selection<'a> {
    pub provider: &'a dyn ModelProvider,
    pub provider_name: &'a str,
    pub model: &'a str,
    pub decision: Option<RouteDecision>,
}

pub trait ModelSource {
    /// # Errors
    /// Returns an error when no usable provider/model can be selected.
    fn next(&mut self, state: &TurnState) -> Result<Selection<'_>, String>;
}

pub struct PinnedModel<'a> {
    provider: &'a dyn ModelProvider,
    model: &'a str,
}

impl<'a> PinnedModel<'a> {
    #[must_use]
    pub const fn new(provider: &'a dyn ModelProvider, model: &'a str) -> Self {
        Self { provider, model }
    }
}

impl ModelSource for PinnedModel<'_> {
    fn next(&mut self, _state: &TurnState) -> Result<Selection<'_>, String> {
        Ok(Selection {
            provider: self.provider,
            provider_name: self.provider.name(),
            model: self.model,
            decision: None,
        })
    }
}

/// UI callbacks for one agent turn. Implementations render streamed text,
/// tool activity, and results; they must not enforce policy.
pub trait TurnObserver {
    fn on_text(&mut self, text: &str);
    fn on_tool_call(&mut self, name: &str, arguments: &str);
    fn on_tool_result(&mut self, name: &str, result: &str);
    fn on_route_changed(&mut self, _decision: &RouteDecision) {}
    /// Line diff of a completed file write. UI-only: it is never added to
    /// the model context.
    fn on_file_diff(&mut self, _path: &str, _diff: &str) {}
}

/// Run the model-driven tool loop until the model stops requesting tools or
/// `max_turns` is exhausted.
///
/// # Errors
///
/// Returns an error when the provider, session recording, or the turn limit
/// fails.
///
/// # Panics
///
/// Never panics in practice: the assistant message it unwraps was pushed
/// onto `messages` immediately beforehand in this same function.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run_loop(
    source: &mut dyn ModelSource,
    workspace: &Workspace,
    tools: &[Value],
    policy: &PolicyEngine,
    messages: &mut Vec<Value>,
    mcp_clients: &mut [McpClient],
    session: &SessionWriter,
    approve: &mut dyn FnMut(&str) -> bool,
    checkpoint: &mut dyn FnMut(&str),
    max_context_chars: usize,
    max_turns: usize,
    cancel: &AtomicBool,
    observer: &mut dyn TurnObserver,
) -> Result<LoopOutcome, String> {
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut consecutive_tool_failures = 0;
    let mut last_provider = String::new();
    let mut last_model = String::new();
    let mut last_band = None;
    let mut switches = 0;
    for turn_index in 0..max_turns {
        if cancel.load(Ordering::Relaxed) {
            session.record("interrupted", "before model turn")?;
            return Ok(LoopOutcome {
                input_tokens,
                output_tokens,
                interrupted: true,
                provider: last_provider,
                model: last_model,
                band: last_band,
                switches,
            });
        }
        let request_messages = context::compact(messages, max_context_chars);
        if request_messages.len() != messages.len() {
            session.record(
                "context_compacted",
                &format!("{} to {} messages", messages.len(), request_messages.len()),
            )?;
        }
        let selection = source.next(&TurnState {
            turn_index,
            turns_remaining: max_turns - turn_index,
            consecutive_tool_failures,
        })?;
        selection.provider_name.clone_into(&mut last_provider);
        selection.model.clone_into(&mut last_model);
        if let Some(decision) = selection.decision.as_ref() {
            let event = if decision.switch {
                "route_changed"
            } else {
                "route_selected"
            };
            session.record(
                event,
                &format!(
                    "{}:{}:{:?}",
                    decision.route.provider, decision.route.model, decision.band
                ),
            )?;
            observer.on_route_changed(decision);
            last_band = Some(format!("{:?}", decision.band).to_lowercase());
            if decision.switch {
                switches += 1;
            }
        }
        let turn =
            selection
                .provider
                .stream_turn(selection.model, &request_messages, tools, cancel)?;
        if turn.tool_calls.is_empty() && !assistant_has_content(&turn.assistant_message) {
            return Err(
                "provider returned an empty assistant turn; history was left unchanged".to_owned(),
            );
        }
        input_tokens = input_tokens.max(turn.input_tokens);
        output_tokens += turn.output_tokens;
        for text in &turn.text_deltas {
            session.record("text_delta", text)?;
            observer.on_text(text);
        }
        messages.push(turn.assistant_message);
        session.record_message(messages.last().expect("assistant message was just pushed"))?;
        if turn.tool_calls.is_empty() {
            let interrupted = cancel.load(Ordering::Relaxed);
            session.record(
                if interrupted {
                    "interrupted"
                } else {
                    "completed"
                },
                &format!("input={input_tokens}, output={output_tokens}"),
            )?;
            return Ok(LoopOutcome {
                input_tokens,
                output_tokens,
                interrupted,
                provider: last_provider,
                model: last_model,
                band: last_band,
                switches,
            });
        }
        for call in turn.tool_calls {
            // A tool result must be recorded for every declared call even
            // after an interrupt, or the next request would be rejected for
            // pairing a dangling tool_calls message.
            // Capture the pre-write content before the tool runs so the
            // observer can show what actually changed.
            let tool_policy = policy.snapshot();
            let unrestricted = tool_policy.unrestricted_access();
            let write_preview = write_preview(workspace, &call, unrestricted);
            let result = if cancel.load(Ordering::Relaxed) {
                "ERROR: interrupted by user".to_owned()
            } else {
                observer.on_tool_call(&call.name, &call.arguments);
                execute_tool(
                    workspace,
                    &call,
                    &tool_policy,
                    approve,
                    checkpoint,
                    mcp_clients,
                )
            };
            session.record("tool_result", &format!("{}: {result}", call.name))?;
            observer.on_tool_result(&call.name, &result);
            if let Some((path, old, new)) = write_preview
                && !result.starts_with("ERROR")
            {
                let rendered = crate::diff::unified(&old, &new);
                if !rendered.is_empty() {
                    observer.on_file_diff(&path, &rendered);
                }
            }
            if result.starts_with("ERROR:") {
                consecutive_tool_failures += 1;
            } else {
                consecutive_tool_failures = 0;
            }
            let tool_message = json!({"role": "tool", "tool_call_id": call.id, "content": result});
            session.record_message(&tool_message)?;
            messages.push(tool_message);
        }
    }
    Err(format!("agent exceeded the {max_turns}-turn safety limit"))
}

fn assistant_has_content(message: &Value) -> bool {
    match message.get("content") {
        Some(Value::String(content)) => !content.is_empty(),
        Some(Value::Array(content)) => !content.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

/// Classify a tool by risk. MCP tools can execute arbitrary server-defined
/// behavior, so they are treated as `Execute` (always requires approval)
/// rather than trusted based on their self-reported description.
fn tool_risk(name: &str) -> Option<ToolRisk> {
    if name.starts_with("mcp__") {
        return Some(ToolRisk::Execute);
    }
    BUILTIN_TOOLS
        .iter()
        .find(|definition| definition.name == name)
        .map(|definition| definition.risk)
}

/// For a `write_file` call, the path plus old and new content, captured
/// before the write so a diff can be shown afterwards. `None` for other
/// tools or unparsable arguments.
fn write_preview(
    workspace: &Workspace,
    call: &ToolCall,
    unrestricted: bool,
) -> Option<(String, String, String)> {
    if call.name != "write_file" {
        return None;
    }
    let arguments: Value = serde_json::from_str(&call.arguments).ok()?;
    let path = arguments.get("path")?.as_str()?.to_owned();
    let new = arguments.get("content")?.as_str()?.to_owned();
    let old = workspace
        .read_file_with_access(Path::new(&path), unrestricted)
        .unwrap_or_default();
    Some((path, old, new))
}

/// Label recorded on the checkpoint taken before a mutating tool runs.
fn checkpoint_label(call: &ToolCall, arguments: &Value, path: &str) -> String {
    let detail = match call.name.as_str() {
        "write_file" => path,
        "run_command" => arguments
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or(""),
        _ => "",
    };
    if detail.is_empty() {
        format!("before {}", call.name)
    } else {
        let detail: String = detail.chars().take(60).collect();
        format!("before {}: {detail}", call.name)
    }
}

fn approval_prompt(
    workspace: &Workspace,
    call: &ToolCall,
    arguments: &Value,
    path: &str,
) -> String {
    if let Some((server, tool)) = call
        .name
        .strip_prefix("mcp__")
        .and_then(|name| name.split_once("__"))
    {
        // Show the exact arguments so approval is informed, capped so a
        // hostile tool call cannot flood the terminal.
        let mut rendered = arguments.to_string();
        if rendered.chars().count() > 400 {
            rendered = rendered.chars().take(400).collect();
            rendered.push('…');
        }
        return format!(
            "Junebug requests MCP tool execution: {server}.{tool}\n  arguments: {rendered}"
        );
    }
    match call.name.as_str() {
        "write_file" => {
            let content = arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            let old = workspace.read_file(Path::new(path)).unwrap_or_default();
            let diff = crate::diff::clip(&crate::diff::unified(&old, content), 40);
            if diff.is_empty() {
                format!(
                    "Junebug requests write access: {path} ({} bytes, no line changes)",
                    content.len()
                )
            } else {
                format!("Junebug requests write access: {path}\n{diff}")
            }
        }
        "run_command" => {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            let warning = if crate::tool::is_dangerous_command(command) {
                "\n  ⚠ WARNING: matches Junebug's destructive/network command patterns"
            } else {
                ""
            };
            format!("Junebug requests command execution in the workspace:\n  {command}{warning}")
        }
        other => format!("Junebug requests approval to run {other}"),
    }
}

/// Dispatch a single tool call, enforcing `policy` before any workspace,
/// process, or MCP side effect. `approve` is only consulted when the policy
/// decision is `Ask`; it must return `false` when approval cannot be
/// obtained (e.g. non-interactive output). `checkpoint` is invoked before
/// any permitted mutating tool runs so the prior state can be rewound; it
/// must never block or fail the tool.
#[must_use]
pub fn execute_tool(
    workspace: &Workspace,
    call: &ToolCall,
    policy: &PolicyEngine,
    approve: &mut dyn FnMut(&str) -> bool,
    checkpoint: &mut dyn FnMut(&str),
    mcp_clients: &mut [McpClient],
) -> String {
    let arguments: Value = match serde_json::from_str(&call.arguments) {
        Ok(arguments) => arguments,
        Err(error) => return format!("ERROR: invalid tool arguments: {error}"),
    };
    let Some(risk) = tool_risk(&call.name) else {
        return format!("ERROR: unknown tool: {}", call.name);
    };
    let path = arguments.get("path").and_then(Value::as_str).unwrap_or("");
    let unrestricted = policy.unrestricted_access();
    match policy.evaluate(risk) {
        Decision::Deny => return "ERROR: denied by permission policy".to_owned(),
        Decision::Ask => {
            if !approve(&approval_prompt(workspace, call, &arguments, path)) {
                return "ERROR: denied by permission policy".to_owned();
            }
        }
        Decision::Allow => {}
    }
    if risk != ToolRisk::Read {
        checkpoint(&checkpoint_label(call, &arguments, path));
    }
    let result = if let Some((server, tool)) = call
        .name
        .strip_prefix("mcp__")
        .and_then(|name| name.split_once("__"))
    {
        match mcp_clients.iter_mut().find(|client| client.name == server) {
            Some(client) => client
                .client
                .call(tool, &arguments)
                .map(|value| value.to_string()),
            None => Err(format!("MCP server is not available: {server}")),
        }
    } else {
        match call.name.as_str() {
            "list_dir" => workspace
                .list_dir_with_access(Path::new(path), unrestricted)
                .map(|entries| entries.join("\n")),
            "read_file" => workspace.read_file_with_access(Path::new(path), unrestricted),
            "search" => workspace.search_at(
                arguments.get("query").and_then(Value::as_str).unwrap_or(""),
                Path::new(arguments.get("path").and_then(Value::as_str).unwrap_or(".")),
                unrestricted,
            ),
            "write_file" => {
                let content = arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                workspace
                    .write_file_with_access(Path::new(path), content, unrestricted)
                    .map(|()| format!("wrote {path} ({} bytes)", content.len()))
            }
            "run_command" => {
                let command = arguments
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let timeout_seconds = arguments
                    .get("timeout_seconds")
                    .and_then(Value::as_u64)
                    .unwrap_or(crate::tool::DEFAULT_COMMAND_TIMEOUT_SECS)
                    .clamp(1, crate::tool::MAX_COMMAND_TIMEOUT_SECS);
                workspace.run_command_with_access(
                    command,
                    unrestricted,
                    std::time::Duration::from_secs(timeout_seconds),
                )
            }
            "git_status" => workspace.git_status_at(Path::new(path), unrestricted),
            "git_diff" => workspace.git_diff_at(Path::new(path), unrestricted),
            _ => Err(format!("unknown tool: {}", call.name)),
        }
    };
    result.unwrap_or_else(|error| format!("ERROR: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{assistant_has_content, execute_tool};
    use crate::PermissionMode;
    use crate::policy::PolicyEngine;
    use crate::provider::ToolCall;
    use crate::tool::Workspace;
    use std::path::PathBuf;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "1".to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    #[test]
    fn unknown_tool_is_rejected_before_any_policy_check() {
        let workspace = Workspace::new(PathBuf::from("."));
        let policy = PolicyEngine::new(PermissionMode::WorkspaceWrite, false);
        let mut approve = |_: &str| true;
        let result = execute_tool(
            &workspace,
            &call("does_not_exist", "{}"),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert!(result.contains("unknown tool"));
    }

    #[test]
    fn empty_assistant_messages_are_not_valid_turns() {
        assert!(!assistant_has_content(
            &serde_json::json!({"role":"assistant","content":null})
        ));
        assert!(!assistant_has_content(
            &serde_json::json!({"role":"assistant","content":""})
        ));
        assert!(assistant_has_content(
            &serde_json::json!({"role":"assistant","content":"done"})
        ));
    }

    #[test]
    fn write_approval_prompt_shows_the_line_diff() {
        let root = std::env::temp_dir().join(format!(
            "junebug-approval-diff-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("workspace");
        std::fs::write(root.join("notes.txt"), "alpha\nbeta\n").expect("seed file");
        let workspace = Workspace::new(root.clone());
        let policy = PolicyEngine::new(PermissionMode::Ask, false);
        let mut prompt_seen = String::new();
        let mut approve = |message: &str| {
            prompt_seen = message.to_owned();
            false
        };
        let result = execute_tool(
            &workspace,
            &call(
                "write_file",
                r#"{"path":"notes.txt","content":"alpha\nBETA\n"}"#,
            ),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert_eq!(result, "ERROR: denied by permission policy");
        assert!(
            prompt_seen.contains("- beta") && prompt_seen.contains("+ BETA"),
            "approval prompt must show what the write changes, got: {prompt_seen}"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mcp_tool_denied_by_policy_never_reaches_dispatch() {
        let workspace = Workspace::new(PathBuf::from("."));
        // Execute-risk tools always require approval; a denial must short
        // circuit before looking up the (here, nonexistent) MCP client.
        let policy = PolicyEngine::new(PermissionMode::WorkspaceWrite, false);
        let mut approve = |_: &str| false;
        let result = execute_tool(
            &workspace,
            &call("mcp__docs__lookup", "{}"),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert_eq!(result, "ERROR: denied by permission policy");
    }

    #[test]
    fn dangerous_command_warns_at_approval_and_denial_blocks_it() {
        let workspace = Workspace::new(PathBuf::from("."));
        let policy = PolicyEngine::new(PermissionMode::Ask, false);
        let mut prompt_seen = String::new();
        let mut approve = |message: &str| {
            prompt_seen = message.to_owned();
            false
        };
        let result = execute_tool(
            &workspace,
            &call("run_command", r#"{"command":"rm -rf important"}"#),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert_eq!(result, "ERROR: denied by permission policy");
        assert!(
            prompt_seen.contains("WARNING"),
            "approval prompt must carry the destructive-pattern warning"
        );
        assert!(prompt_seen.contains("rm -rf important"));
    }

    #[test]
    fn read_only_permission_blocks_write_without_prompting() {
        let workspace = Workspace::new(PathBuf::from("."));
        let policy = PolicyEngine::new(PermissionMode::ReadOnly, false);
        let mut prompted = false;
        let mut approve = |_: &str| {
            prompted = true;
            true
        };
        let result = execute_tool(
            &workspace,
            &call("write_file", r#"{"path":"x.txt","content":"hi"}"#),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert_eq!(result, "ERROR: denied by permission policy");
        assert!(!prompted, "read-only denial must not prompt for approval");
    }

    #[test]
    fn yolo_dispatch_reads_protected_and_absolute_paths_without_prompting() {
        let root = std::env::temp_dir().join(format!(
            "junebug-yolo-dispatch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let outside = root.with_extension("outside");
        std::fs::create_dir_all(&root).expect("workspace");
        std::fs::create_dir_all(&outside).expect("outside");
        std::fs::write(root.join(".env"), "LOCAL_SECRET=test-only").expect("protected file");
        std::fs::write(outside.join("note.txt"), "outside").expect("outside file");
        assert!(
            std::process::Command::new("git")
                .args(["init", "--quiet"])
                .arg(&outside)
                .status()
                .expect("git init")
                .success()
        );
        let workspace = Workspace::new(root.clone());
        let policy = PolicyEngine::new(PermissionMode::Yolo, false);
        let mut prompted = false;
        let mut approve = |_: &str| {
            prompted = true;
            false
        };

        let protected = execute_tool(
            &workspace,
            &call("read_file", r#"{"path":".env"}"#),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        let absolute = execute_tool(
            &workspace,
            &call(
                "read_file",
                &serde_json::json!({"path": outside.join("note.txt")}).to_string(),
            ),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        let git_status = execute_tool(
            &workspace,
            &call(
                "git_status",
                &serde_json::json!({"path": &outside}).to_string(),
            ),
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert_eq!(protected, "LOCAL_SECRET=test-only");
        assert_eq!(absolute, "outside");
        assert!(git_status.contains("note.txt"), "got: {git_status}");
        assert!(!prompted);

        std::fs::remove_dir_all(root).expect("cleanup root");
        std::fs::remove_dir_all(outside).expect("cleanup outside");
    }
}
