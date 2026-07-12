//! Explicitly trusted lifecycle hooks loaded from `.febo/hooks.json`.

use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;

/// # Errors
///
/// Returns an error when a hook configuration cannot be read or validated.
pub fn load(workspace: &Path, event: &str) -> Result<Vec<String>, String> {
    let path = workspace.join(".febo").join("hooks.json");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let config: Value =
        serde_json::from_str(&fs::read_to_string(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let hooks = config
        .get(event)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("hooks.json event '{event}' must be an array"))?;
    hooks
        .iter()
        .map(|hook| {
            hook.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("hooks.json event '{event}' contains a non-string command"))
        })
        .collect()
}

/// # Errors
///
/// Returns an error when the command cannot start or exits unsuccessfully.
pub fn run(workspace: &Path, command: &str) -> Result<(), String> {
    let mut process = if cfg!(windows) {
        let mut process = Command::new("cmd");
        process.args(["/C", command]);
        process
    } else {
        let mut process = Command::new("/bin/sh");
        process.args(["-c", command]);
        process
    };
    let output = process
        .current_dir(workspace)
        .env_clear()
        .env("PATH", default_path())
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "hook exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

const fn default_path() -> &'static str {
    if cfg!(windows) {
        "C:\\Windows\\System32"
    } else {
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    }
}

#[cfg(test)]
mod tests {
    use super::load;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn loads_event_commands() {
        let root = std::env::temp_dir().join(format!(
            "febo-hooks-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(root.join(".febo")).expect("directory");
        fs::write(
            root.join(".febo/hooks.json"),
            r#"{"session_start":["echo ok"]}"#,
        )
        .expect("config");
        assert_eq!(load(&root, "session_start").expect("load"), vec!["echo ok"]);
        fs::remove_dir_all(root).expect("cleanup");
    }
}
