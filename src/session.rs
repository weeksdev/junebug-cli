//! Append-only local session recording and structured resume support.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub struct SessionWriter {
    path: PathBuf,
}

impl SessionWriter {
    /// # Errors
    ///
    /// Returns an error when the local session directory cannot be created.
    pub fn create(workspace: &Path) -> Result<Self, String> {
        let directory = workspace.join(".febo").join("sessions");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis();
        Ok(Self {
            path: directory.join(format!("{millis}.jsonl")),
        })
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
    use super::{SessionWriter, load_messages};
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn records_and_loads_messages() {
        let root = std::env::temp_dir().join(format!(
            "febo-session-test-{}",
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
