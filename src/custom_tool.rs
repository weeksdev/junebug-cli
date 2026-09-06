//! Custom tools: a single named capability backed by an external
//! executable, *produced and tested* conversationally via `/tool-build`
//! rather than hand-authored — the small-grained sibling of `plugin.rs`
//! (which delegates a whole conversational turn) and `PLUGIN_PROTOCOL.md`
//! (whose contract this deliberately does not reuse — a tool call carries
//! structured arguments, not a prompt). See `CUSTOM_TOOL_PROTOCOL.md` for
//! the full wire contract.
//!
//! Stored as one JSON manifest per tool, at `~/.junebug/tools/<name>.json`
//! (global) and `.junebug/tools/<name>.json` (repo) — both loaded and
//! merged, the repo copy winning a name collision, the same pattern
//! `custom_agent.rs`/`commands.rs` already use.
//!
//! A custom tool is always classified `ToolRisk::Execute` (it can do
//! anything its script does) — the same treatment MCP tools already get —
//! so outside `yolo` it always requires an explicit approval, and it never
//! appears in plan mode's tool set at all.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Upper bound on one custom-tool call. Generous but finite, same
/// reasoning as `run_command`'s own timeout and `cli_delegate`'s
/// `DELEGATE_TIMEOUT`: a hang must still end without a human present to
/// interrupt it.
const CALL_TIMEOUT: Duration = Duration::from_secs(crate::tool::MAX_COMMAND_TIMEOUT_SECS);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Global,
    Repo,
}

impl Scope {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Repo => "repo",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomTool {
    pub name: String,
    pub description: String,
    /// A JSON Schema `object` — the same shape a builtin tool's
    /// `function.parameters` already uses (see `main.rs::tool_schemas`).
    pub parameters: Value,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

impl CustomTool {
    /// The `{"type":"function","function":{...}}` shape every provider
    /// expects, identical to how a builtin tool is declared in
    /// `main.rs::tool_schemas`.
    #[must_use]
    pub fn schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

pub struct ToolEntry {
    pub tool: CustomTool,
    pub scope: Scope,
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn dir_for(scope: Scope, workspace: &Path) -> Option<PathBuf> {
    match scope {
        Scope::Global => home().map(|home| home.join(".junebug").join("tools")),
        Scope::Repo => Some(workspace.join(".junebug").join("tools")),
    }
}

/// Every configured tool, global and repo merged — a repo tool overrides a
/// global one of the same name.
#[must_use]
pub fn list(workspace: &Path) -> Vec<ToolEntry> {
    merge(&[
        (dir_for(Scope::Global, workspace), Scope::Global),
        (dir_for(Scope::Repo, workspace), Scope::Repo),
    ])
}

/// Pure merge over already-resolved directories — see `custom_agent::merge`
/// for why this is split out (testable without touching real `HOME`).
fn merge(dirs: &[(Option<PathBuf>, Scope)]) -> Vec<ToolEntry> {
    let mut by_name: BTreeMap<String, ToolEntry> = BTreeMap::new();
    for (dir, scope) in dirs {
        let Some(dir) = dir else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Ok(contents) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(tool) = serde_json::from_str::<CustomTool>(&contents) else {
                continue;
            };
            by_name.insert(
                tool.name.clone(),
                ToolEntry {
                    tool,
                    scope: *scope,
                },
            );
        }
    }
    by_name.into_values().collect()
}

#[must_use]
pub fn load(workspace: &Path, name: &str) -> Option<ToolEntry> {
    list(workspace)
        .into_iter()
        .find(|entry| entry.tool.name == name)
}

/// # Errors
///
/// Returns an error when the target scope's directory cannot be created or
/// the file cannot be written.
pub fn save(workspace: &Path, tool: &CustomTool, scope: Scope) -> Result<PathBuf, String> {
    let dir = dir_for(scope, workspace).ok_or("cannot locate a home directory")?;
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join(format!("{}.json", tool.name));
    let contents = serde_json::to_string_pretty(tool).map_err(|error| error.to_string())?;
    std::fs::write(&path, contents).map_err(|error| error.to_string())?;
    Ok(path)
}

/// # Errors
///
/// Returns an error when the file does not exist in that scope or cannot be
/// removed.
pub fn delete(workspace: &Path, name: &str, scope: Scope) -> Result<(), String> {
    let dir = dir_for(scope, workspace).ok_or("cannot locate a home directory")?;
    std::fs::remove_file(dir.join(format!("{name}.json"))).map_err(|error| error.to_string())
}

/// Every configured tool's `{"type":"function",...}` schema, for splicing
/// into the live tool list a turn is offered.
#[must_use]
pub fn schemas(workspace: &Path) -> Vec<Value> {
    list(workspace)
        .iter()
        .map(|entry| entry.tool.schema())
        .collect()
}

// ---------------------------------------------------------------------
// Execution: one JSON object each way over stdin/stdout — see
// CUSTOM_TOOL_PROTOCOL.md for the full contract.
// ---------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawResponse {
    #[serde(default)]
    result: String,
    #[serde(default)]
    is_error: bool,
}

/// Run one call to a custom tool. Never panics; a broken tool becomes an
/// `ERROR:`-prefixed result string, the same convention every builtin tool
/// failure already uses (see `agent::execute_tool`).
#[must_use]
pub fn call(tool: &CustomTool, arguments: &Value, workspace: &Path) -> String {
    let request = json!({
        "arguments": arguments,
        "workspace": workspace.display().to_string(),
    });

    let mut command = Command::new(&tool.command);
    command.args(&tool.args);
    command.current_dir(workspace);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return format!(
                "ERROR: {}",
                describe_spawn_error(&tool.name, &tool.command, &error)
            );
        }
    };

    {
        use std::io::Write as _;
        let Some(mut stdin) = child.stdin.take() else {
            return format!("ERROR: could not open stdin for tool '{}'", tool.name);
        };
        if let Err(error) = stdin.write_all(request.to_string().as_bytes()) {
            return format!("ERROR: {error}");
        }
        // `stdin` drops here, closing the pipe so the tool sees EOF.
    }

    let stdout = crate::tool::drain_stream(child.stdout.take());
    let stderr = crate::tool::drain_stream(child.stderr.take());
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Err(error) => return format!("ERROR: {error}"),
            Ok(Some(_)) => break,
            Ok(None) => {}
        }
        if started.elapsed() >= CALL_TIMEOUT {
            crate::tool::kill_command_tree(&mut child);
            let _ = child.wait();
            return format!(
                "ERROR: tool '{}' timed out after {}s and was killed",
                tool.name,
                CALL_TIMEOUT.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(30));
    }

    let (stdout_text, _) = crate::tool::collect_one(&stdout);
    let (stderr_text, _) = crate::tool::collect_one(&stderr);
    let parsed: RawResponse = match serde_json::from_str(stdout_text.trim()) {
        Ok(parsed) => parsed,
        Err(error) => {
            let tail = tail_chars(
                if stderr_text.trim().is_empty() {
                    &stdout_text
                } else {
                    &stderr_text
                },
                2000,
            );
            return if tail.is_empty() {
                format!(
                    "ERROR: tool '{}' did not return a valid response: {error}",
                    tool.name
                )
            } else {
                format!(
                    "ERROR: tool '{}' did not return a valid response ({error}):\n{tail}",
                    tool.name
                )
            };
        }
    };
    if parsed.is_error {
        format!("ERROR: {}", parsed.result)
    } else {
        parsed.result
    }
}

fn describe_spawn_error(name: &str, command: &str, error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        format!("tool '{name}' points at '{command}', which does not exist or is not executable")
    } else {
        format!("cannot run tool '{name}' ({command}): {error}")
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

// ---------------------------------------------------------------------
// Conversational builder.
// ---------------------------------------------------------------------

pub const TOOL_BUILDER_SYSTEM: &str = "You are helping design and build a new custom tool for \
Junebug, a coding-agent CLI. A tool is a small executable script you will write and test \
yourself using your write_file and run_command tools, then register. Have a real back-and-forth \
first: what should this tool actually do, what arguments does it need, what should it return? \
Then write the script (any language available in this environment — a shebang line makes it \
directly executable; make it executable with chmod +x via run_command). The script MUST speak \
this exact protocol: read one JSON object from stdin shaped {\"arguments\": <the call's \
arguments, matching your proposed parameters schema>, \"workspace\": \"<absolute path>\"}, and \
write exactly one JSON object to stdout shaped {\"result\": \"<text result>\", \"is_error\": \
<bool>} — nothing else on stdout (use stderr for any debug output). Before proposing it as done, \
actually TEST it: run the script directly via run_command, piping in a realistic sample request \
matching the protocol above, and confirm the stdout is exactly one valid JSON object in that \
shape. If it fails, fix it and test again — do not propose a tool you have not verified. Once \
verified, reply with exactly one ```json fenced object with fields: name (short snake_case \
identifier — this becomes the tool's callable name, so it must look like a function name, not a \
sentence), description (what it does and when to use it, written for a model deciding whether to \
call it), parameters (a JSON Schema object, the same shape a function-calling tool's parameters \
schema always takes — e.g. {\"type\":\"object\",\"properties\":{...},\"required\":[...]}), \
command (the absolute path to the script you just wrote and tested), and args (array of any \
fixed extra arguments — usually empty). End that final message with exactly one line: TOOL: \
READY";

#[must_use]
pub fn build_request(notes: &str) -> String {
    format!(
        "Task notes: {notes}\n\nDesign and build a custom tool for this. Ask whatever you need \
         to know first, then write and actually test the script before proposing it as done."
    )
}

#[must_use]
pub fn update_request(existing: &CustomTool, notes: &str) -> String {
    format!(
        "You are refining an EXISTING custom tool, not creating one from scratch. Current \
         definition:\nname: {}\ndescription: {}\nparameters: {}\ncommand: {}\nargs: {:?}\n\n\
         Requested changes: {notes}\n\nInspect the existing script (read_file on `command`), \
         make the change, test it again the same way as before, then reply with the FULL \
         updated spec (not a diff) in the same ```json object + TOOL: READY format.",
        existing.name, existing.description, existing.parameters, existing.command, existing.args,
    )
}

/// Strict last-non-blank-line check, mirroring `swarm::parse_verdict`.
#[must_use]
pub fn parse_ready(text: &str) -> bool {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim().eq_ignore_ascii_case("TOOL: READY"))
}

#[derive(Debug, Deserialize)]
struct RawSpec {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: Option<Value>,
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

/// Extract the builder's proposed tool spec from its reply.
///
/// # Errors
///
/// Returns an error when no parsable spec object is present, its name is
/// unusable, or `command` is empty.
pub fn parse_spec(text: &str) -> Result<CustomTool, String> {
    let raw =
        crate::jsonblock::find_object(text).ok_or("the reply contains no JSON tool spec object")?;
    let spec: RawSpec = serde_json::from_str(raw)
        .map_err(|error| format!("could not parse the tool spec: {error}"))?;
    let name = crate::custom_agent::slugify(&spec.name).replace('-', "_");
    if name.is_empty() {
        return Err("the tool spec's name is empty once normalized".to_owned());
    }
    if spec.command.trim().is_empty() {
        return Err("the tool spec has an empty command".to_owned());
    }
    Ok(CustomTool {
        name,
        description: if spec.description.trim().is_empty() {
            "custom tool".to_owned()
        } else {
            spec.description
        },
        parameters: spec.parameters.unwrap_or_else(
            || json!({"type": "object", "properties": {}, "additionalProperties": true}),
        ),
        command: spec.command,
        args: spec.args,
    })
}

/// A deterministic one-line-per-tool listing for `/tools` — no model call.
#[must_use]
pub fn format_list(entries: &[ToolEntry]) -> String {
    use std::fmt::Write as _;
    if entries.is_empty() {
        return "no custom tools configured — build one with /tool-build <notes>".to_owned();
    }
    let mut out = String::new();
    for entry in entries {
        let _ = writeln!(
            out,
            "  {:<20} [{}] {}",
            entry.tool.name,
            entry.scope.label(),
            entry.tool.description
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "junebug-custom-tool-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ))
    }

    fn sample(name: &str) -> CustomTool {
        CustomTool {
            name: name.to_owned(),
            description: "counts widgets".to_owned(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            command: "/usr/bin/true".to_owned(),
            args: vec![],
        }
    }

    #[test]
    fn schema_matches_the_builtin_function_tool_shape() {
        let tool = sample("count_widgets");
        let schema = tool.schema();
        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "count_widgets");
        assert_eq!(schema["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn parses_a_fenced_tool_spec_and_normalizes_the_name() {
        let reply = "Tested it, works.\n```json\n{\"name\":\"Count Widgets\",\"command\":\"/tmp/count.py\"}\n```\nTOOL: READY";
        assert!(parse_ready(reply));
        let tool = parse_spec(reply).expect("tool");
        assert_eq!(tool.name, "count_widgets");
        assert_eq!(tool.description, "custom tool");
        assert_eq!(tool.parameters["type"], "object");
    }

    #[test]
    fn empty_command_is_rejected() {
        let reply = "```json\n{\"name\":\"a\",\"command\":\"\"}\n```";
        assert!(parse_spec(reply).is_err());
    }

    #[test]
    fn ready_signal_is_strict() {
        assert!(!parse_ready("TOOL: READY\nbut wait"));
        assert!(parse_ready("some text\nTOOL: READY"));
    }

    #[test]
    fn save_load_list_and_delete_round_trip_within_repo_scope() {
        let workspace = scratch_dir("roundtrip");
        std::fs::create_dir_all(&workspace).expect("workspace");

        let tool = sample("count_widgets");
        save(&workspace, &tool, Scope::Repo).expect("save");
        assert_eq!(list(&workspace).len(), 1);
        let loaded = load(&workspace, "count_widgets").expect("loaded");
        assert_eq!(loaded.tool.description, "counts widgets");
        assert_eq!(schemas(&workspace).len(), 1);

        delete(&workspace, "count_widgets", Scope::Repo).expect("delete");
        assert!(list(&workspace).is_empty());

        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn merge_lets_repo_override_global_on_name_collision() {
        let global_dir = scratch_dir("merge-global");
        let repo_dir = scratch_dir("merge-repo");
        std::fs::create_dir_all(&global_dir).expect("global dir");
        std::fs::create_dir_all(&repo_dir).expect("repo dir");

        let write = |dir: &Path, tool: &CustomTool| {
            std::fs::write(
                dir.join(format!("{}.json", tool.name)),
                serde_json::to_string_pretty(tool).expect("serialize"),
            )
            .expect("write");
        };
        write(
            &global_dir,
            &CustomTool {
                description: "global version".to_owned(),
                ..sample("shared_name")
            },
        );
        write(
            &repo_dir,
            &CustomTool {
                description: "repo version".to_owned(),
                ..sample("shared_name")
            },
        );

        let entries = merge(&[
            (Some(global_dir.clone()), Scope::Global),
            (Some(repo_dir.clone()), Scope::Repo),
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tool.description, "repo version");

        std::fs::remove_dir_all(&global_dir).expect("cleanup global");
        std::fs::remove_dir_all(&repo_dir).expect("cleanup repo");
    }

    #[test]
    fn call_round_trips_through_a_real_script() {
        let workspace = scratch_dir("call");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let script = workspace.join("echo_tool.py");
        std::fs::write(
            &script,
            "#!/usr/bin/env python3\nimport json, sys\nreq = json.load(sys.stdin)\nprint(json.dumps({\"result\": \"got: \" + str(req[\"arguments\"]), \"is_error\": False}))\n",
        )
        .expect("write script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }
        let tool = CustomTool {
            name: "echo_tool".to_owned(),
            description: "echoes its arguments".to_owned(),
            parameters: json!({"type": "object", "properties": {}}),
            command: "python3".to_owned(),
            args: vec![script.display().to_string()],
        };
        let result = call(&tool, &json!({"x": 1}), &workspace);
        assert!(result.contains("got:"), "unexpected result: {result}");
        assert!(result.contains('1'));

        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }
}
