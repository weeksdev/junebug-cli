//! Integration fixtures proving that the agent loop never performs a write
//! or command that the `PolicyEngine` did not allow, using a scripted
//! `ModelProvider` and a disposable temporary workspace — no network access
//! or real model is involved.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use febo_cli::PermissionMode;
use febo_cli::agent::{self, TurnObserver};
use febo_cli::policy::PolicyEngine;
use febo_cli::provider::{ModelProvider, ModelTurn, ToolCall};
use febo_cli::session::SessionWriter;
use febo_cli::tool::Workspace;
use serde_json::{Value, json};

/// Observer that discards all UI events.
struct Silent;

impl TurnObserver for Silent {
    fn on_text(&mut self, _: &str) {}
    fn on_tool_call(&mut self, _: &str, _: &str) {}
    fn on_tool_result(&mut self, _: &str, _: &str) {}
}

/// A `ModelProvider` that replays a fixed script of turns instead of calling
/// a real API. Each call to `stream_turn` consumes the next scripted turn.
struct FixtureProvider {
    turns: RefCell<VecDeque<ModelTurn>>,
}

impl FixtureProvider {
    fn new(turns: Vec<ModelTurn>) -> Self {
        Self {
            turns: RefCell::new(turns.into()),
        }
    }
}

impl ModelProvider for FixtureProvider {
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn stream_turn(
        &self,
        _messages: &[Value],
        _tools: &[Value],
        _cancel: &AtomicBool,
    ) -> Result<ModelTurn, String> {
        self.turns
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| "fixture provider script exhausted".to_owned())
    }
}

fn write_file_turn(path: &str, content: &str) -> ModelTurn {
    let arguments = json!({"path": path, "content": content}).to_string();
    ModelTurn {
        text_deltas: Vec::new(),
        tool_calls: vec![ToolCall {
            id: "call-1".to_owned(),
            name: "write_file".to_owned(),
            arguments: arguments.clone(),
        }],
        assistant_message: json!({"role": "assistant", "content": Value::Null, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}]}),
        input_tokens: 10,
        output_tokens: 5,
    }
}

fn final_turn() -> ModelTurn {
    ModelTurn {
        text_deltas: vec!["done".to_owned()],
        tool_calls: Vec::new(),
        assistant_message: json!({"role": "assistant", "content": "done"}),
        input_tokens: 1,
        output_tokens: 1,
    }
}

fn temp_workspace(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "febo-policy-integration-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&path).expect("create temp workspace");
    path
}

#[test]
fn read_only_permission_blocks_write_end_to_end() {
    let root = temp_workspace("read-only");
    let workspace = Workspace::new(root.clone());
    let session = SessionWriter::create(&root).expect("session");
    let provider = FixtureProvider::new(vec![write_file_turn("notes.txt", "hello"), final_turn()]);
    let policy = PolicyEngine::new(PermissionMode::ReadOnly, false);
    let mut messages = vec![json!({"role": "user", "content": "write a file"})];
    let mut mcp_clients: Vec<agent::McpClient> = Vec::new();
    let mut approve = |_: &str| true; // must never be consulted: ReadOnly denies before asking

    agent::run_loop(
        &provider,
        &workspace,
        &[],
        &policy,
        &mut messages,
        &mut mcp_clients,
        &session,
        &mut approve,
        100_000,
        5,
        &AtomicBool::new(false),
        &mut Silent,
    )
    .expect("loop completes");

    assert!(
        !root.join("notes.txt").exists(),
        "write must not reach the filesystem under read-only permission"
    );
    let tool_message = messages
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool result message recorded");
    assert!(
        tool_message["content"]
            .as_str()
            .expect("string content")
            .contains("denied"),
        "tool result must record the denial"
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn workspace_write_permission_allows_write_end_to_end() {
    let root = temp_workspace("workspace-write");
    let workspace = Workspace::new(root.clone());
    let session = SessionWriter::create(&root).expect("session");
    let provider = FixtureProvider::new(vec![write_file_turn("notes.txt", "hello"), final_turn()]);
    let policy = PolicyEngine::new(PermissionMode::WorkspaceWrite, false);
    let mut messages = vec![json!({"role": "user", "content": "write a file"})];
    let mut mcp_clients: Vec<agent::McpClient> = Vec::new();
    let mut approve = |_: &str| false; // must never be consulted: WorkspaceWrite pre-approves

    agent::run_loop(
        &provider,
        &workspace,
        &[],
        &policy,
        &mut messages,
        &mut mcp_clients,
        &session,
        &mut approve,
        100_000,
        5,
        &AtomicBool::new(false),
        &mut Silent,
    )
    .expect("loop completes");

    assert_eq!(
        fs::read_to_string(root.join("notes.txt")).expect("file was written"),
        "hello"
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn ask_permission_only_writes_when_approved() {
    let root = temp_workspace("ask-denied");
    let workspace = Workspace::new(root.clone());
    let session = SessionWriter::create(&root).expect("session");
    let provider = FixtureProvider::new(vec![write_file_turn("notes.txt", "hello"), final_turn()]);
    let policy = PolicyEngine::new(PermissionMode::Ask, false);
    let mut messages = vec![json!({"role": "user", "content": "write a file"})];
    let mut mcp_clients: Vec<agent::McpClient> = Vec::new();
    let mut approve_calls = 0;
    let mut approve = |_: &str| {
        approve_calls += 1;
        false
    };

    agent::run_loop(
        &provider,
        &workspace,
        &[],
        &policy,
        &mut messages,
        &mut mcp_clients,
        &session,
        &mut approve,
        100_000,
        5,
        &AtomicBool::new(false),
        &mut Silent,
    )
    .expect("loop completes");

    assert_eq!(
        approve_calls, 1,
        "ask mode must consult the approver exactly once"
    );
    assert!(!root.join("notes.txt").exists());

    fs::remove_dir_all(root).expect("cleanup");
}
