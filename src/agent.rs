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
use crate::session::SessionWriter;
use crate::tool::{BUILTIN_TOOLS, ToolRisk, Workspace};

pub struct McpClient {
    pub name: String,
    pub client: mcp::Client,
}

#[derive(Debug, Clone, Copy)]
pub struct LoopOutcome {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub interrupted: bool,
}

/// UI callbacks for one agent turn. Implementations render streamed text,
/// tool activity, and results; they must not enforce policy.
pub trait TurnObserver {
    fn on_text(&mut self, text: &str);
    fn on_tool_call(&mut self, name: &str, arguments: &str);
    fn on_tool_result(&mut self, name: &str, result: &str);
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
#[allow(clippy::too_many_arguments)]
pub fn run_loop(
    provider: &dyn ModelProvider,
    workspace: &Workspace,
    tools: &[Value],
    policy: &PolicyEngine,
    messages: &mut Vec<Value>,
    mcp_clients: &mut [McpClient],
    session: &SessionWriter,
    approve: &mut dyn FnMut(&str) -> bool,
    max_context_chars: usize,
    max_turns: u8,
    cancel: &AtomicBool,
    observer: &mut dyn TurnObserver,
) -> Result<LoopOutcome, String> {
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for _ in 0..max_turns {
        if cancel.load(Ordering::Relaxed) {
            session.record("interrupted", "before model turn")?;
            return Ok(LoopOutcome {
                input_tokens,
                output_tokens,
                interrupted: true,
            });
        }
        let request_messages = context::compact(messages, max_context_chars);
        if request_messages.len() != messages.len() {
            session.record(
                "context_compacted",
                &format!("{} to {} messages", messages.len(), request_messages.len()),
            )?;
        }
        let turn = provider.stream_turn(&request_messages, tools, cancel)?;
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
            });
        }
        for call in turn.tool_calls {
            // A tool result must be recorded for every declared call even
            // after an interrupt, or the next request would be rejected for
            // pairing a dangling tool_calls message.
            let result = if cancel.load(Ordering::Relaxed) {
                "ERROR: interrupted by user".to_owned()
            } else {
                observer.on_tool_call(&call.name, &call.arguments);
                execute_tool(workspace, &call, policy, approve, mcp_clients)
            };
            session.record("tool_result", &format!("{}: {result}", call.name))?;
            observer.on_tool_result(&call.name, &result);
            let tool_message = json!({"role": "tool", "tool_call_id": call.id, "content": result});
            session.record_message(&tool_message)?;
            messages.push(tool_message);
        }
    }
    Err(format!("agent exceeded the {max_turns}-turn safety limit"))
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

fn approval_prompt(call: &ToolCall, arguments: &Value, path: &str) -> String {
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
            "Febo requests MCP tool execution: {server}.{tool}\n  arguments: {rendered}"
        );
    }
    match call.name.as_str() {
        "write_file" => {
            let bytes = arguments
                .get("content")
                .and_then(Value::as_str)
                .map_or(0, str::len);
            format!("Febo requests write access: {path} ({bytes} bytes)")
        }
        "run_command" => {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            let warning = if crate::tool::is_dangerous_command(command) {
                "\n  ⚠ WARNING: matches Febo's destructive/network command patterns"
            } else {
                ""
            };
            format!("Febo requests command execution in the workspace:\n  {command}{warning}")
        }
        other => format!("Febo requests approval to run {other}"),
    }
}

/// Dispatch a single tool call, enforcing `policy` before any workspace,
/// process, or MCP side effect. `approve` is only consulted when the policy
/// decision is `Ask`; it must return `false` when approval cannot be
/// obtained (e.g. non-interactive output).
#[must_use]
pub fn execute_tool(
    workspace: &Workspace,
    call: &ToolCall,
    policy: &PolicyEngine,
    approve: &mut dyn FnMut(&str) -> bool,
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
    match policy.evaluate(risk) {
        Decision::Deny => return "ERROR: denied by permission policy".to_owned(),
        Decision::Ask => {
            if !approve(&approval_prompt(call, &arguments, path)) {
                return "ERROR: denied by permission policy".to_owned();
            }
        }
        Decision::Allow => {}
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
                .list_dir(Path::new(path))
                .map(|entries| entries.join("\n")),
            "read_file" => workspace.read_file(Path::new(path)),
            "search" => {
                workspace.search(arguments.get("query").and_then(Value::as_str).unwrap_or(""))
            }
            "write_file" => {
                let content = arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                workspace
                    .write_file(Path::new(path), content)
                    .map(|()| format!("wrote {path} ({} bytes)", content.len()))
            }
            "run_command" => {
                let command = arguments
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                workspace.run_command(command)
            }
            "git_status" => workspace.git_status(),
            "git_diff" => workspace.git_diff(),
            _ => Err(format!("unknown tool: {}", call.name)),
        }
    };
    result.unwrap_or_else(|error| format!("ERROR: {error}"))
}

#[cfg(test)]
mod tests {
    use super::execute_tool;
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
            &mut [],
        );
        assert!(result.contains("unknown tool"));
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
            &mut [],
        );
        assert_eq!(result, "ERROR: denied by permission policy");
        assert!(!prompted, "read-only denial must not prompt for approval");
    }
}
