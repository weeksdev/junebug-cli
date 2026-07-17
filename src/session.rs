//! Append-only local session recording and structured resume support.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

/// Distinguishes sessions created in the same millisecond by one process
/// (e.g. concurrent swarm workers).
static SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub struct SessionWriter {
    path: PathBuf,
}

impl SessionWriter {
    /// # Errors
    ///
    /// Returns an error when the local session directory or a unique session
    /// file cannot be created.
    pub fn create(workspace: &Path) -> Result<Self, String> {
        let directory = workspace.join(".junebug").join("sessions");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis();
        let pid = std::process::id();
        // Timestamp + pid + sequence is unique across concurrent processes
        // and rapid creation within one; `create_new` closes what remains
        // (pid reuse, clock skew) by refusing to adopt an existing file, so
        // two sessions can never interleave into one log.
        let mut last_error = String::new();
        for _ in 0..16 {
            let sequence = SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!("{millis}-{pid}-{sequence}.jsonl"));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    last_error = error.to_string();
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Err(format!(
            "could not create a unique session file: {last_error}"
        ))
    }

    /// # Errors
    ///
    /// Returns an error when `path` does not name a session file.
    pub fn open(path: PathBuf) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("session does not exist: {}", path.display()));
        }
        Ok(Self { path })
    }

    /// # Errors
    ///
    /// Returns an error when the event cannot be appended to the session log.
    pub fn record(&self, event: &str, value: &str) -> Result<(), String> {
        self.append(&json!({"event": event, "value": value}))
    }

    /// Persist a provider-compatible conversation message for later resume.
    ///
    /// # Errors
    ///
    /// Returns an error when the message cannot be appended to the session log.
    pub fn record_message(&self, message: &Value) -> Result<(), String> {
        self.append(&json!({"event": "message", "message": message}))
    }

    fn append(&self, event: &Value) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        writeln!(
            file,
            "{}",
            serde_json::to_string(&event).expect("JSON values serialize")
        )
        .map_err(|error| error.to_string())
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// A summary of a recorded session, for the resume picker.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub path: PathBuf,
    /// Modification time, used to sort newest first.
    pub modified: SystemTime,
    /// First user prompt in the session, for a human-readable preview.
    pub preview: String,
    /// Number of conversation messages recorded.
    pub messages: usize,
    /// Provider name recorded for the session, if any.
    pub provider: Option<String>,
    /// Model in effect at the end of the session (for the final provider).
    pub model: Option<String>,
}

/// The provider name recorded in the most recent session for this workspace,
/// used to default `--provider` to the last one used.
#[must_use]
pub fn last_provider(workspace: &Path) -> Option<String> {
    list_sessions(workspace)
        .ok()?
        .into_iter()
        .find_map(|summary| summary.provider)
}

/// The model in effect at the end of the most recent session, so the model
/// choice is sticky across runs like the provider. Returns `None` when that
/// session ended on a different provider (model names are provider-specific).
#[must_use]
pub fn last_model(workspace: &Path, provider: &str) -> Option<String> {
    let summary = list_sessions(workspace)
        .ok()?
        .into_iter()
        .find(|summary| summary.provider.is_some())?;
    if summary.provider.as_deref() == Some(provider) {
        summary.model
    } else {
        None
    }
}

/// List recorded sessions under `.junebug/sessions` and the legacy
/// `.febo/sessions`, newest first. Unreadable or empty files are skipped
/// rather than failing the whole listing.
///
/// # Errors
///
/// Returns an error only when the sessions directory exists but cannot be
/// read.
pub fn list_sessions(workspace: &Path) -> Result<Vec<SessionSummary>, String> {
    let mut summaries = Vec::new();
    for directory in [
        workspace.join(".junebug").join("sessions"),
        workspace.join(".febo").join("sessions"),
    ] {
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(summary) = summarize_session(&path) else {
                continue;
            };
            summaries.push(summary);
        }
    }
    summaries.sort_by_key(|summary| std::cmp::Reverse(summary.modified));
    Ok(summaries)
}

/// The last `max_chars` of the newest session log that contains a
/// `swarm_goal` event — the raw material for an AI progress summary of a
/// running or aborted swarm. `None` when no swarm log exists.
#[must_use]
pub fn latest_swarm_log_tail(workspace: &Path, max_chars: usize) -> Option<String> {
    let mut newest: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for directory in [
        workspace.join(".junebug").join("sessions"),
        workspace.join(".febo").join("sessions"),
    ] {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
                continue;
            };
            if newest.as_ref().is_some_and(|(when, _)| *when >= modified) {
                continue;
            }
            let Ok(head) = fs::read_to_string(&path) else {
                continue;
            };
            if head.contains("\"swarm_goal\"") {
                newest = Some((modified, path));
            }
        }
    }
    let (_, path) = newest?;
    let contents = fs::read_to_string(path).ok()?;
    let count = contents.chars().count();
    Some(
        contents
            .chars()
            .skip(count.saturating_sub(max_chars))
            .collect(),
    )
}

fn summarize_session(path: &Path) -> Option<SessionSummary> {
    let file = File::open(path).ok()?;
    let modified = path.metadata().and_then(|meta| meta.modified()).ok()?;
    let mut preview = String::new();
    let mut messages = 0usize;
    let mut provider = None;
    let mut model = None;
    let mut standalone_swarm_log = false;
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let Ok(event) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match event.get("event").and_then(Value::as_str) {
            Some("message") => {
                messages += 1;
                // Older logs and swarm summaries may have structured user
                // messages without the redundant `user_prompt` event. Use
                // the first such message as a picker title fallback.
                if preview.is_empty()
                    && let Some(message) = event.get("message")
                    && message.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(content) = message.get("content").and_then(Value::as_str)
                {
                    first_preview_line(content).clone_into(&mut preview);
                }
            }
            Some("provider") => {
                provider = event
                    .get("value")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                // A mid-session provider switch invalidates any model seen so
                // far: model names are provider-specific.
                model = None;
            }
            Some("model" | "model_changed") => {
                model = event
                    .get("value")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            Some("swarm_goal") => {
                // A swarm session is an audit log containing several fresh
                // boss/worker/checker histories concatenated together. It is
                // intentionally not a provider-resumable conversation and
                // must not appear in the `--resume` picker.
                standalone_swarm_log = true;
            }
            Some("user_prompt") if preview.is_empty() => {
                if let Some(value) = event.get("value").and_then(Value::as_str) {
                    first_preview_line(value).clone_into(&mut preview);
                }
            }
            _ => {}
        }
    }
    if standalone_swarm_log || (messages == 0 && preview.is_empty()) {
        return None;
    }
    Some(SessionSummary {
        path: path.to_path_buf(),
        modified,
        preview,
        messages,
        provider,
        model,
    })
}

fn first_preview_line(value: &str) -> &str {
    value.lines().next().unwrap_or("").trim()
}

/// Load structured conversation messages from an append-only session log.
///
/// # Errors
///
/// Returns an error when the session cannot be read or contains malformed JSON.
pub fn load_messages(path: &Path) -> Result<Vec<Value>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut messages = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if event.get("event").and_then(Value::as_str) == Some("message") {
            messages.push(
                event
                    .get("message")
                    .cloned()
                    .ok_or("session message event lacks message")?,
            );
        }
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::{SessionWriter, last_model, list_sessions, load_messages};
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn model_is_sticky_for_the_matching_provider() {
        let root = std::env::temp_dir().join(format!(
            "junebug-last-model-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let session = SessionWriter::create(&root).expect("session");
        session.record("provider", "deepseek").expect("provider");
        session.record("model", "deepseek-v4-flash").expect("model");
        session.record("user_prompt", "task").expect("prompt");
        session
            .record("model_changed", "deepseek-v4-pro")
            .expect("switch");

        assert_eq!(
            last_model(&root, "deepseek").as_deref(),
            Some("deepseek-v4-pro"),
            "the model in effect at session end must win"
        );
        assert_eq!(
            last_model(&root, "openai"),
            None,
            "a model must never leak to a different provider"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn mid_session_provider_switch_resets_the_sticky_model() {
        let root = std::env::temp_dir().join(format!(
            "junebug-provider-switch-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let session = SessionWriter::create(&root).expect("session");
        session.record("provider", "deepseek").expect("provider");
        session
            .record("model_changed", "deepseek-v4-pro")
            .expect("switch");
        session.record("user_prompt", "task").expect("prompt");
        session
            .record("provider", "openai")
            .expect("provider switch");

        assert_eq!(
            last_model(&root, "openai"),
            None,
            "a provider switch without a model choice must not carry the old model"
        );
        assert_eq!(last_model(&root, "deepseek"), None);

        session.record("model_changed", "gpt-5.4").expect("switch");
        assert_eq!(last_model(&root, "openai").as_deref(), Some("gpt-5.4"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn lists_sessions_newest_first_with_preview() {
        let root = std::env::temp_dir().join(format!(
            "junebug-list-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        assert!(list_sessions(&root).expect("empty ok").is_empty());

        let older = SessionWriter::create(&root).expect("older session");
        older.record("user_prompt", "first task").expect("prompt");
        older
            .record_message(&json!({"role": "user", "content": "first task"}))
            .expect("message");
        // Ensure a distinct, newer mtime for the second session.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let newer = SessionWriter::create(&root).expect("newer session");
        newer.record("user_prompt", "second task").expect("prompt");

        let sessions = list_sessions(&root).expect("list");
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].preview, "second task");
        assert_eq!(sessions[1].preview, "first task");
        assert_eq!(sessions[1].messages, 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn legacy_febo_sessions_remain_listed() {
        let root = std::env::temp_dir().join(format!(
            "junebug-legacy-session-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let directory = root.join(".febo/sessions");
        fs::create_dir_all(&directory).expect("legacy directory");
        fs::write(
            directory.join("old.jsonl"),
            "{\"event\":\"user_prompt\",\"value\":\"legacy task\"}\n",
        )
        .expect("legacy session");

        let sessions = list_sessions(&root).expect("list");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].preview, "legacy task");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn message_only_session_uses_first_user_message_as_preview() {
        let root = std::env::temp_dir().join(format!(
            "junebug-message-preview-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let session = SessionWriter::create(&root).expect("session");
        session
            .record_message(&json!({"role": "assistant", "content": "not the title"}))
            .expect("assistant message");
        session
            .record_message(&json!({"role": "user", "content": "swarm goal\nmore detail"}))
            .expect("user message");

        let sessions = list_sessions(&root).expect("list");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].preview, "swarm goal");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn standalone_swarm_log_is_not_offered_for_resume() {
        let root = std::env::temp_dir().join(format!(
            "junebug-swarm-preview-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let session = SessionWriter::create(&root).expect("session");
        session
            .record("swarm_goal", "do the next phase where we left off")
            .expect("goal");
        session
            .record_message(&json!({"role": "assistant", "content": null, "tool_calls": []}))
            .expect("assistant message");

        assert!(
            list_sessions(&root).expect("list").is_empty(),
            "multi-agent swarm audit histories are not resumable conversations"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn concurrent_session_creation_never_reuses_a_path() {
        let root = std::env::temp_dir().join(format!(
            "junebug-session-unique-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let root = root.clone();
                std::thread::spawn(move || {
                    (0..32)
                        .map(|_| {
                            SessionWriter::create(&root)
                                .expect("session")
                                .path()
                                .to_path_buf()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut seen = std::collections::HashSet::new();
        for handle in handles {
            for path in handle.join().expect("thread") {
                assert!(
                    seen.insert(path.clone()),
                    "duplicate session path: {}",
                    path.display()
                );
                assert!(path.is_file(), "session file must exist on creation");
            }
        }
        assert_eq!(seen.len(), 256);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn records_and_loads_messages() {
        let root = std::env::temp_dir().join(format!(
            "junebug-session-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("directory");
        let session = SessionWriter::create(&root).expect("session");
        session
            .record_message(&json!({"role": "user", "content": "hello"}))
            .expect("write");
        assert_eq!(
            load_messages(session.path()).expect("load"),
            vec![json!({"role": "user", "content": "hello"})]
        );
        fs::remove_dir_all(root).expect("cleanup");
    }
}
