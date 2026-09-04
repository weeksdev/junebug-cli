//! Generic external-agent plugin provider (`--provider plugin --model
//! <name>`, or `/model plugin:<name>`).
//!
//! `cli_delegate` hardcodes two specific external agents (`claude`, `codex`)
//! and speaks each one's own CLI flags/output format. This module is the
//! same idea generalized: a plugin is any local executable, named by a
//! small JSON manifest, that speaks one fixed protocol Junebug defines
//! itself — so any external agent (in particular, one you write yourself
//! wrapping the Claude Agent SDK against your own subscription) can be
//! wired in without Junebug's own code ever touching that agent's
//! authentication.
//!
//! This exists specifically so a personal, **never-distributed** companion
//! program can use the Claude Agent SDK directly against your own Pro/Max
//! subscription (via `claude setup-token`'s `CLAUDE_CODE_OAUTH_TOKEN`)
//! without any of that code — or credential — ever living in this
//! open-source repository. Junebug only ever knows "run this configured
//! executable, speak this protocol"; it has no idea the executable happens
//! to use the Agent SDK, and ships with zero OAuth-handling code. See
//! `PLUGIN_PROTOCOL.md` for the manifest format and wire protocol a plugin
//! must implement.
//!
//! Like `cli_delegate`, a plugin runs its own complete agent loop
//! internally: `stream_turn` always returns zero `tool_calls`, and the
//! `tools` argument is ignored. Junebug's permission mode is passed through
//! as a plain string in the request so the plugin can map it onto its own
//! sandbox, exactly as `cli_delegate::effective_sandbox` does for
//! `claude`/`codex`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::PermissionMode;
use crate::cli_delegate::{latest_user_text, tail_chars};
use crate::provider::{ModelProvider, ModelTurn};

/// Upper bound on one plugin turn — the plugin runs its own agentic loop
/// with no way for Junebug to see progress, so a hang must still end
/// eventually even without a user Esc. Same value as `cli_delegate`'s
/// `DELEGATE_TIMEOUT`, for the same reason.
const PLUGIN_TIMEOUT: Duration = Duration::from_mins(10);

#[derive(Debug, Deserialize)]
struct Manifest {
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

/// `.junebug/plugins/<name>.json` in the workspace, else
/// `~/.junebug/plugins/<name>.json`. Mirrors the lookup order
/// `swarm.rs`/`abduction.rs` use for their own role-config files.
fn manifest_path(workspace: &Path, name: &str) -> Option<PathBuf> {
    let workspace_file = workspace
        .join(".junebug")
        .join("plugins")
        .join(format!("{name}.json"));
    if workspace_file.is_file() {
        return Some(workspace_file);
    }
    let home_dir = home()?;
    let home_file = home_dir
        .join(".junebug")
        .join("plugins")
        .join(format!("{name}.json"));
    home_file.is_file().then_some(home_file)
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn load_manifest(workspace: &Path, name: &str) -> Result<Manifest, String> {
    let path = manifest_path(workspace, name).ok_or_else(|| {
        format!(
            "no plugin named '{name}' — create ~/.junebug/plugins/{name}.json with \
             {{\"command\": \"/path/to/executable\"}} (workspace .junebug/plugins/{name}.json \
             overrides it)"
        )
    })?;
    let contents = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| format!("{}: {error}", path.display()))
}

/// Every configured plugin name discoverable from the workspace and
/// user-level plugin directories, sorted and deduplicated. Used by the
/// `/model plugin` picker; `pub(crate)` because `main.rs` calls it directly
/// rather than going through `ModelProvider::list_models` (which needs a
/// constructed `PluginProvider`, i.e. an already-chosen name).
#[must_use]
pub fn list_available_names(workspace: &Path) -> Vec<String> {
    let mut names = Vec::new();
    for dir in [workspace.join(".junebug").join("plugins")]
        .into_iter()
        .chain(home().map(|home_dir| home_dir.join(".junebug").join("plugins")))
    {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                names.push(stem.to_owned());
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Whether at least one plugin is configured at the user level. `has_credential`
/// on `ProviderKind::Plugin` has no workspace path to check (it takes no
/// argument), so this only ever sees `~/.junebug/plugins/` — a workspace-only
/// plugin still works once selected, it just won't make the provider show up
/// as "available" from a bare `junebug` launch in that workspace.
#[must_use]
pub fn any_plugin_configured() -> bool {
    let Some(home_dir) = home() else {
        return false;
    };
    std::fs::read_dir(home_dir.join(".junebug").join("plugins")).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
    })
}

pub struct PluginProvider {
    name: String,
    workspace: PathBuf,
    permission: Mutex<PermissionMode>,
}

impl PluginProvider {
    /// # Errors
    ///
    /// Returns an error when no plugin name was given, or the named
    /// plugin's manifest cannot be found or parsed.
    pub fn new(
        name: Option<String>,
        workspace: PathBuf,
        permission: PermissionMode,
    ) -> Result<Self, String> {
        let name = name.filter(|value| !value.is_empty()).ok_or_else(|| {
            "a plugin provider needs a name — use --model <plugin-name> or /model plugin:<name>"
                .to_owned()
        })?;
        // Fail fast on a typo'd name at startup rather than on the first
        // turn; the manifest is re-read on every turn after this (see
        // `stream_turn`) so editing it live never needs a restart.
        load_manifest(&workspace, &name)?;
        Ok(Self {
            name,
            workspace,
            permission: Mutex::new(permission),
        })
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.name
    }

    /// Switching the plugin name is just switching which manifest
    /// `stream_turn` reads next turn — unlike `CliDelegateProvider`, there's
    /// no in-memory session id to invalidate (a plugin owns its own
    /// continuity, if any, entirely on its side of the protocol).
    pub fn set_model(&mut self, name: String) {
        self.name = name;
    }

    pub fn set_permission(&self, permission: PermissionMode) {
        if let Ok(mut guard) = self.permission.lock() {
            *guard = permission;
        }
    }

    /// Every plugin manifest visible from this provider's workspace —
    /// unlike `CliDelegateProvider::list_models` (which always errors),
    /// plugins have a real discoverable local catalog.
    ///
    /// # Errors
    ///
    /// Never actually errors; `Result` only to match the shape every other
    /// provider's `list_models` uses.
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        Ok(list_available_names(&self.workspace))
    }
}

#[derive(Debug, Deserialize)]
struct PluginResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    is_error: bool,
}

impl ModelProvider for PluginProvider {
    fn name(&self) -> &'static str {
        "plugin"
    }

    #[allow(clippy::too_many_lines)]
    fn stream_turn(
        &self,
        _model: &str,
        messages: &[Value],
        _tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<ModelTurn, String> {
        let manifest = load_manifest(&self.workspace, &self.name)?;
        let prompt = latest_user_text(messages)
            .ok_or_else(|| "no user message to send to the plugin".to_owned())?;
        let permission = *self
            .permission
            .lock()
            .map_err(|_| "permission lock poisoned".to_owned())?;
        let request = json!({
            "prompt": prompt,
            "permission": permission.as_str(),
            "workspace": self.workspace.display().to_string(),
        });

        let mut command = Command::new(&manifest.command);
        command.args(&manifest.args);
        command.current_dir(&self.workspace);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .map_err(|error| describe_spawn_error(&self.name, &manifest.command, &error))?;

        {
            use std::io::Write as _;
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| format!("could not open stdin for plugin '{}'", self.name))?;
            stdin
                .write_all(request.to_string().as_bytes())
                .map_err(|error| error.to_string())?;
            // `stdin` drops here, closing the pipe so the plugin sees EOF.
        }

        let stdout = crate::tool::drain_stream(child.stdout.take());
        let stderr = crate::tool::drain_stream(child.stderr.take());
        let started = Instant::now();
        loop {
            match child.try_wait() {
                Err(error) => return Err(error.to_string()),
                Ok(Some(_)) => break,
                Ok(None) => {}
            }
            if cancel.load(Ordering::Relaxed) {
                crate::tool::kill_command_tree(&mut child);
                let _ = child.wait();
                return Err("plugin turn cancelled by user".to_owned());
            }
            if started.elapsed() >= PLUGIN_TIMEOUT {
                crate::tool::kill_command_tree(&mut child);
                let _ = child.wait();
                return Err(format!(
                    "plugin '{}' timed out after {}s and was killed",
                    self.name,
                    PLUGIN_TIMEOUT.as_secs()
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        let (stdout_text, _) = crate::tool::collect_one(&stdout);
        let (stderr_text, _) = crate::tool::collect_one(&stderr);
        let parsed: PluginResponse = serde_json::from_str(stdout_text.trim()).map_err(|error| {
            let tail = tail_chars(
                if stderr_text.trim().is_empty() {
                    &stdout_text
                } else {
                    &stderr_text
                },
                2000,
            );
            if tail.is_empty() {
                format!(
                    "plugin '{}' did not return a valid response: {error}",
                    self.name
                )
            } else {
                format!(
                    "plugin '{}' did not return a valid response ({error}):\n{tail}",
                    self.name
                )
            }
        })?;

        if parsed.is_error {
            return Err(if parsed.text.trim().is_empty() {
                format!("plugin '{}' reported an error", self.name)
            } else {
                parsed.text
            });
        }

        let assistant_message = json!({
            "role": "assistant",
            "content": if parsed.text.is_empty() { Value::Null } else { Value::String(parsed.text.clone()) },
        });
        Ok(ModelTurn {
            text_deltas: vec![parsed.text],
            tool_calls: Vec::new(),
            assistant_message,
            input_tokens: parsed.input_tokens,
            output_tokens: parsed.output_tokens,
        })
    }
}

fn describe_spawn_error(name: &str, command: &str, error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        format!("plugin '{name}' points at '{command}', which does not exist or is not executable")
    } else {
        format!("cannot run plugin '{name}' ({command}): {error}")
    }
}

#[cfg(test)]
mod tests {
    use super::{list_available_names, load_manifest};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "junebug-plugin-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ))
    }

    #[test]
    fn missing_plugin_reports_a_helpful_error() {
        let workspace = scratch_dir("missing");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let error = load_manifest(&workspace, "nope").unwrap_err();
        assert!(error.contains("no plugin named 'nope'"));
        assert!(error.contains("~/.junebug/plugins/nope.json"));
        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn workspace_manifest_is_found_and_parsed() {
        let workspace = scratch_dir("workspace-manifest");
        let dir = workspace.join(".junebug").join("plugins");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(
            dir.join("bridge.json"),
            r#"{"command": "/usr/bin/true", "args": ["--flag"]}"#,
        )
        .expect("write");
        let manifest = load_manifest(&workspace, "bridge").expect("manifest");
        assert_eq!(manifest.command, "/usr/bin/true");
        assert_eq!(manifest.args, vec!["--flag".to_owned()]);
        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn list_available_names_is_sorted_and_deduplicated() {
        let workspace = scratch_dir("listing");
        let dir = workspace.join(".junebug").join("plugins");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("zeta.json"), r#"{"command":"/bin/true"}"#).expect("write");
        std::fs::write(dir.join("alpha.json"), r#"{"command":"/bin/true"}"#).expect("write");
        std::fs::write(dir.join("notes.txt"), "ignored").expect("write");
        let names = list_available_names(&workspace);
        assert!(names.contains(&"alpha".to_owned()));
        assert!(names.contains(&"zeta".to_owned()));
        assert!(!names.contains(&"notes".to_owned()));
        let alpha_index = names.iter().position(|name| name == "alpha").unwrap();
        let zeta_index = names.iter().position(|name| name == "zeta").unwrap();
        assert!(alpha_index < zeta_index);
        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }
}
