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

/// `run_loop`'s error when a provider turn has no tool calls and no text
/// after exhausting the empty-turn retry budget.
const EMPTY_TURN_ERROR: &str =
    "provider returned an empty assistant turn; history was left unchanged";

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
    /// An out-of-band status line (e.g. a retry after a provider error).
    /// UI-only, like diffs.
    fn on_notice(&mut self, _text: &str) {}
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
    // Whole-loop retry budgets, matching the swarm's: transient stream
    // deaths retry fast, rate limits wait the window out.
    let mut transient_retries = 0usize;
    let mut rate_limit_retries = 0usize;
    let mut empty_turn_retries = 0usize;
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
            // This safety-net trim used to be invisible — the only trace was
            // the session-log line above, so it could silently rewrite
            // history (and, before a related fix, occasionally produce a
            // malformed request) with no on-screen sign anything happened.
            // Claude Code shows compaction live; this is the same idea.
            observer.on_notice(&format!(
                "context compacted: {} → {} messages",
                messages.len(),
                request_messages.len()
            ));
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
        let turn = loop {
            match selection
                .provider
                .stream_turn(selection.model, &request_messages, tools, cancel)
            {
                // Some routed/niche models (seen via OpenRouter) occasionally
                // return a turn with no tool calls and no text at all — a
                // one-off upstream hiccup, not a real "the model is done"
                // signal. Retry it exactly like a transient transport error:
                // nothing was added to `messages` or the session yet, so a
                // retry is safe.
                Ok(turn)
                    if turn.tool_calls.is_empty()
                        && !assistant_has_content(&turn.assistant_message) =>
                {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(EMPTY_TURN_ERROR.to_owned());
                    }
                    let Some(&delay) = crate::swarm::TRANSIENT_DELAYS.get(empty_turn_retries)
                    else {
                        return Err(EMPTY_TURN_ERROR.to_owned());
                    };
                    empty_turn_retries += 1;
                    session.record(
                        "turn_retry",
                        &format!("retrying in {delay}s: {EMPTY_TURN_ERROR}"),
                    )?;
                    observer.on_notice(&format!(
                        "provider returned an empty turn — retrying in {delay}s"
                    ));
                    let deadline =
                        std::time::Instant::now() + std::time::Duration::from_secs(delay);
                    while std::time::Instant::now() < deadline {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(EMPTY_TURN_ERROR.to_owned());
                        }
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                }
                Ok(turn) => break turn,
                Err(error) => {
                    // A user interrupt surfaces as a provider error; never
                    // retry it. Retrying a whole turn is safe: nothing was
                    // added to `messages` or the session on failure.
                    if cancel.load(Ordering::Relaxed) {
                        return Err(error);
                    }
                    let Some(delay) = crate::swarm::retry_delay(
                        &error,
                        &mut transient_retries,
                        &mut rate_limit_retries,
                    ) else {
                        return Err(error);
                    };
                    session.record("turn_retry", &format!("retrying in {delay}s: {error}"))?;
                    observer.on_notice(&format!("provider error — retrying in {delay}s ({error})"));
                    // Sliced sleep so Esc keeps working during the wait.
                    let deadline =
                        std::time::Instant::now() + std::time::Duration::from_secs(delay);
                    while std::time::Instant::now() < deadline {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(error);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                }
            }
        };
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
            } else if call.name == "task" {
                observer.on_tool_call(&call.name, &call.arguments);
                run_subagent(
                    workspace,
                    selection.provider,
                    selection.model,
                    &call.arguments,
                    tools,
                    &tool_policy,
                    mcp_clients,
                    approve,
                    checkpoint,
                    max_context_chars,
                    max_turns,
                    cancel,
                    observer,
                )
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
    let arguments: Value = serde_json::from_str(&call.arguments).ok()?;
    let path = arguments.get("path")?.as_str()?.to_owned();
    let old = workspace
        .read_file_with_access(Path::new(&path), unrestricted)
        .unwrap_or_default();
    let new = match call.name.as_str() {
        "write_file" => arguments.get("content")?.as_str()?.to_owned(),
        "edit_file" => {
            let (updated, _) = planned_edit(&old, &arguments)?;
            updated
        }
        _ => return None,
    };
    Some((path, old, new))
}

/// The post-edit contents an `edit_file` call would produce, from its
/// arguments — shared by the approval diff and the post-run diff event.
fn planned_edit(current: &str, arguments: &Value) -> Option<(String, usize)> {
    crate::tool::edit_outcome(
        current,
        arguments.get("old_text").and_then(Value::as_str)?,
        arguments.get("new_text").and_then(Value::as_str)?,
        arguments
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )
    .ok()
}

/// Label recorded on the checkpoint taken before a mutating tool runs.
fn checkpoint_label(call: &ToolCall, arguments: &Value, path: &str) -> String {
    let detail = match call.name.as_str() {
        "write_file" | "edit_file" => path,
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
        "web_search" => {
            // The query text leaves the machine; show exactly what is sent
            // so the approval is informed.
            let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
            format!("Junebug requests a web search (the query is sent to DuckDuckGo):\n  {query}")
        }
        "fetch_url" => {
            let url = arguments.get("url").and_then(Value::as_str).unwrap_or("");
            format!("Junebug requests to fetch a URL over the network:\n  {url}")
        }
        "edit_file" => {
            let old = workspace.read_file(Path::new(path)).unwrap_or_default();
            match planned_edit(&old, arguments) {
                Some((updated, _)) => {
                    let diff = crate::diff::clip(&crate::diff::unified(&old, &updated), 40);
                    format!("Junebug requests edit access: {path}\n{diff}")
                }
                // The edit will fail (no match, ambiguous); still show what
                // was asked so a decline is informed.
                None => format!("Junebug requests edit access: {path} (match preview unavailable)"),
            }
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
#[allow(clippy::too_many_lines)]
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
            "read_file" => {
                let offset = arguments.get("offset").and_then(Value::as_u64);
                let limit = arguments.get("limit").and_then(Value::as_u64);
                if offset.is_some() || limit.is_some() {
                    workspace.read_file_slice_with_access(
                        Path::new(path),
                        usize::try_from(offset.unwrap_or(1)).unwrap_or(1),
                        usize::try_from(limit.unwrap_or(2000)).unwrap_or(2000),
                        unrestricted,
                    )
                } else {
                    workspace.read_file_with_access(Path::new(path), unrestricted)
                }
            }
            "search" => workspace.search_at(
                arguments.get("query").and_then(Value::as_str).unwrap_or(""),
                Path::new(arguments.get("path").and_then(Value::as_str).unwrap_or(".")),
                unrestricted,
            ),
            "semantic_search" => {
                let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
                let limit = arguments
                    .get("limit")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(crate::semsearch::DEFAULT_RESULTS);
                crate::semsearch::search(workspace.root(), query, limit)
            }
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
            "web_search" => {
                let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
                let max_results = arguments
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(crate::websearch::DEFAULT_RESULTS);
                crate::websearch::web_search(query, max_results)
            }
            "fetch_url" => {
                let url = arguments.get("url").and_then(Value::as_str).unwrap_or("");
                let max_chars = arguments
                    .get("max_chars")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(crate::webfetch::DEFAULT_MAX_CHARS);
                crate::webfetch::fetch_url(url, max_chars)
            }
            "edit_file" => {
                let old_text = arguments
                    .get("old_text")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let new_text = arguments
                    .get("new_text")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let replace_all = arguments
                    .get("replace_all")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                workspace.edit_file_with_access(
                    Path::new(path),
                    old_text,
                    new_text,
                    replace_all,
                    unrestricted,
                )
            }
            "write_todos" => {
                let todos = arguments
                    .get("todos")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| {
                                let content =
                                    item.get("content").and_then(Value::as_str)?.to_owned();
                                let status = match item
                                    .get("status")
                                    .and_then(Value::as_str)
                                    .unwrap_or("pending")
                                {
                                    "in_progress" => crate::tool::TodoStatus::InProgress,
                                    "completed" => crate::tool::TodoStatus::Completed,
                                    _ => crate::tool::TodoStatus::Pending,
                                };
                                Some(crate::tool::Todo { content, status })
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                Ok(workspace.set_todos(todos))
            }
            _ => Err(format!("unknown tool: {}", call.name)),
        }
    };
    result.unwrap_or_else(|error| format!("ERROR: {error}"))
}

/// Forwards a sub-agent's activity into the parent turn's observer as
/// out-of-band notices instead of `on_text`/`on_tool_*`, so its intermediate
/// steps show up as a quiet log line rather than interleaving with the
/// parent's own streamed reply. File diffs still pass through directly —
/// a sub-agent editing a file is real workspace activity worth showing
/// exactly like the parent's own edits.
struct SubagentRelay<'a> {
    inner: &'a mut dyn TurnObserver,
    label: &'a str,
}

impl TurnObserver for SubagentRelay<'_> {
    fn on_text(&mut self, _text: &str) {}

    fn on_tool_call(&mut self, name: &str, arguments: &str) {
        self.inner.on_notice(&format!(
            "↳ {}: {name}({})",
            self.label,
            clip(arguments, 60)
        ));
    }

    fn on_tool_result(&mut self, _name: &str, result: &str) {
        self.inner
            .on_notice(&format!("↳ {}: → {}", self.label, clip(result, 80)));
    }

    fn on_file_diff(&mut self, path: &str, diff: &str) {
        self.inner.on_file_diff(path, diff);
    }
}

/// First line of `text`, truncated to `max` characters — enough to make an
/// activity notice legible without echoing a sub-agent's full output.
fn clip(text: &str, max: usize) -> String {
    let first_line = text.lines().next().unwrap_or("").trim();
    if first_line.chars().count() <= max {
        first_line.to_owned()
    } else {
        let mut truncated: String = first_line.chars().take(max).collect();
        truncated.push('…');
        truncated
    }
}

/// Dispatch a `task` call: run a nested, single-purpose agent loop with a
/// fresh, isolated message history and its own session log, returning only
/// its final answer to the caller. This is the "context quarantine" half of
/// the deep-agent pattern — the parent never sees the sub-agent's
/// intermediate tool calls, only a clean summary, so delegating a large
/// investigation doesn't bloat the parent's own context. Sub-agents cannot
/// spawn further sub-agents or touch the shared todo list (both are
/// excluded from their tool list), so delegation is exactly one level deep
/// and the plan stays owned by whichever agent is actually showing it to
/// the user.
#[allow(clippy::too_many_arguments)]
fn run_subagent(
    workspace: &Workspace,
    provider: &dyn ModelProvider,
    model: &str,
    arguments: &str,
    tools: &[Value],
    policy: &PolicyEngine,
    mcp_clients: &mut [McpClient],
    approve: &mut dyn FnMut(&str) -> bool,
    checkpoint: &mut dyn FnMut(&str),
    max_context_chars: usize,
    max_turns: usize,
    cancel: &AtomicBool,
    observer: &mut dyn TurnObserver,
) -> String {
    let arguments: Value = match serde_json::from_str(arguments) {
        Ok(arguments) => arguments,
        Err(error) => return format!("ERROR: invalid tool arguments: {error}"),
    };
    let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
        return "ERROR: task requires a \"prompt\" argument".to_owned();
    };
    let label = arguments
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("sub-agent")
        .to_owned();
    let sub_tools: Vec<Value> = tools
        .iter()
        .filter(|tool| {
            !matches!(
                tool.pointer("/function/name").and_then(Value::as_str),
                Some("task" | "write_todos")
            )
        })
        .cloned()
        .collect();
    let session = match SessionWriter::create(workspace.root()) {
        Ok(session) => session,
        Err(error) => return format!("ERROR: could not start sub-agent session: {error}"),
    };
    let mut messages = vec![
        json!({
            "role": "system",
            "content": format!(
                "You are a focused sub-agent spawned to complete one self-contained task and report back. There is no user watching this conversation live and you cannot ask follow-up questions, so do the best you can with what you were given, then finish with a clear, complete final answer summarizing what you found or did. The startup workspace is exactly: {}",
                workspace.root().display()
            )
        }),
        json!({"role": "user", "content": prompt}),
    ];
    let mut source = PinnedModel::new(provider, model);
    let mut relay = SubagentRelay {
        inner: observer,
        label: &label,
    };
    if let Err(error) = run_loop(
        &mut source,
        workspace,
        &sub_tools,
        policy,
        &mut messages,
        mcp_clients,
        &session,
        approve,
        checkpoint,
        max_context_chars,
        max_turns,
        cancel,
        &mut relay,
    ) {
        return format!("ERROR: sub-agent failed: {error}");
    }
    messages
        .last()
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map_or_else(
            || "(sub-agent finished with no final message)".to_owned(),
            ToOwned::to_owned,
        )
}

#[cfg(test)]
mod tests {
    use super::{PinnedModel, TurnObserver, assistant_has_content, execute_tool, run_loop};
    use crate::PermissionMode;
    use crate::policy::PolicyEngine;
    use crate::provider::{ModelProvider, ModelTurn, ToolCall};
    use crate::session::SessionWriter;
    use crate::tool::Workspace;
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

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
    fn web_search_is_gated_by_network_policy_before_any_request() {
        let workspace = Workspace::new(PathBuf::from("."));
        let request = call("web_search", r#"{"query":"junebug cli"}"#);
        // Plan mode hard-denies network risk without consulting approval.
        let policy = PolicyEngine::new(PermissionMode::Yolo, true);
        let mut approve = |_: &str| panic!("plan mode must never ask for approval");
        let result = execute_tool(
            &workspace,
            &request,
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert!(result.contains("denied"), "got: {result}");
        // Outside yolo the approval prompt shows the outgoing query, and a
        // declined approval blocks the call before any network activity.
        let policy = PolicyEngine::new(PermissionMode::WorkspaceWrite, false);
        let mut approve = |prompt: &str| {
            assert!(prompt.contains("DuckDuckGo"), "got: {prompt}");
            assert!(prompt.contains("junebug cli"), "got: {prompt}");
            false
        };
        let result = execute_tool(
            &workspace,
            &request,
            &policy,
            &mut approve,
            &mut |_: &str| {},
            &mut [],
        );
        assert!(result.contains("denied"), "got: {result}");
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

    /// A provider that returns one empty turn (no tool calls, no content)
    /// before answering normally — simulating a one-off empty completion
    /// from a flaky upstream route.
    struct FlakyProvider {
        calls: Cell<usize>,
    }

    impl ModelProvider for FlakyProvider {
        fn name(&self) -> &'static str {
            "flaky"
        }

        fn stream_turn(
            &self,
            _model: &str,
            _messages: &[serde_json::Value],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
        ) -> Result<ModelTurn, String> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            let assistant_message = if call == 0 {
                serde_json::json!({"role": "assistant", "content": null})
            } else {
                serde_json::json!({"role": "assistant", "content": "done"})
            };
            Ok(ModelTurn {
                text_deltas: if call == 0 {
                    vec![]
                } else {
                    vec!["done".to_owned()]
                },
                tool_calls: vec![],
                assistant_message,
                input_tokens: 1,
                output_tokens: 1,
            })
        }
    }

    struct SilentObserver;

    impl TurnObserver for SilentObserver {
        fn on_text(&mut self, _text: &str) {}
        fn on_tool_call(&mut self, _name: &str, _arguments: &str) {}
        fn on_tool_result(&mut self, _name: &str, _result: &str) {}
    }

    #[test]
    fn empty_provider_turns_are_retried_instead_of_failing_the_run() {
        let root = std::env::temp_dir().join(format!(
            "junebug-empty-turn-retry-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("workspace");
        let workspace = Workspace::new(root.clone());
        let policy = PolicyEngine::new(PermissionMode::ReadOnly, false);
        let provider = FlakyProvider {
            calls: Cell::new(0),
        };
        let mut source = PinnedModel::new(&provider, "flaky-model");
        let session = SessionWriter::create(workspace.root()).expect("session");
        let mut messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let mut approve = |_: &str| true;
        let mut checkpoint = |_: &str| {};
        let cancel = AtomicBool::new(false);
        let mut observer = SilentObserver;
        let outcome = run_loop(
            &mut source,
            &workspace,
            &[],
            &policy,
            &mut messages,
            &mut [],
            &session,
            &mut approve,
            &mut checkpoint,
            100_000,
            4,
            &cancel,
            &mut observer,
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
        let outcome = outcome.expect("an empty turn should be retried, not fatal");
        assert_eq!(
            provider.calls.get(),
            2,
            "should retry once after the empty turn"
        );
        assert_eq!(outcome.output_tokens, 1);
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
