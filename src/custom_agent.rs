//! Custom agents: a user-defined system prompt plus tool subset (and an
//! optional tighter permission cap), built *conversationally* via
//! `/agent-build` (or refined via `/agent-update`) rather than hand-authored.
//!
//! Stored as one JSON file per agent, at `~/.junebug/agents/<name>.json`
//! (global) and `.junebug/agents/<name>.json` (repo) — both are loaded and
//! merged, the repo copy winning a name collision, exactly mirroring
//! `commands.rs`'s load-and-merge for custom slash commands.
//!
//! A custom agent's tools/permission can only ever *narrow* what the
//! ambient permission mode already allows, never widen it — the same
//! principle the `task` tool already documents (a sub-agent runs "with the
//! same tools and permissions" as its caller). See `main.rs`'s
//! `run_named_agent_turn` (direct `/<name>` invocation, one turn appended to
//! the shared conversation) and `agent::run_subagent`'s `agent` parameter
//! (invocation from the main agent's own `task` tool call, seeded with the
//! caller's history) for the two ways one actually runs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PermissionMode;

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
pub struct CustomAgent {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub tools: Vec<String>,
    /// A permission-mode name (`"read-only"`, `"ask"`, `"workspace-write"`,
    /// `"yolo"`) acting as a cap tighter than the session's ambient
    /// permission — e.g. a "verify"/"smoke test" agent that should never
    /// write files even under `yolo`. Stored as a plain string (not
    /// `PermissionMode` itself, which has no serde impl) and resolved with
    /// `effective_permission`. Never used to grant more than the ambient
    /// mode allows.
    #[serde(default)]
    pub permission: Option<String>,
}

impl CustomAgent {
    /// The permission this agent should actually run under: its own cap if
    /// one is set and stricter than `ambient`, else `ambient` unchanged.
    /// Never returns something looser than `ambient` — a cap only narrows.
    #[must_use]
    pub fn effective_permission(&self, ambient: PermissionMode) -> PermissionMode {
        let Some(cap) = self
            .permission
            .as_deref()
            .and_then(|value| PermissionMode::parse(value).ok())
        else {
            return ambient;
        };
        if permission_rank(cap) < permission_rank(ambient) {
            cap
        } else {
            ambient
        }
    }
}

const fn permission_rank(permission: PermissionMode) -> u8 {
    match permission {
        PermissionMode::ReadOnly => 0,
        PermissionMode::Ask => 1,
        PermissionMode::WorkspaceWrite => 2,
        PermissionMode::Yolo => 3,
    }
}

pub struct AgentEntry {
    pub agent: CustomAgent,
    pub scope: Scope,
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn dir_for(scope: Scope, workspace: &Path) -> Option<PathBuf> {
    match scope {
        Scope::Global => home().map(|home| home.join(".junebug").join("agents")),
        Scope::Repo => Some(workspace.join(".junebug").join("agents")),
    }
}

/// Every configured agent, global and repo merged — a repo agent overrides
/// a global one of the same name. Unreadable or unparsable files are
/// skipped rather than failing startup, same as `commands.rs`.
#[must_use]
pub fn list(workspace: &Path) -> Vec<AgentEntry> {
    // Repo listed second, so it overrides a global entry of the same name
    // in `merge`.
    merge(&[
        (dir_for(Scope::Global, workspace), Scope::Global),
        (dir_for(Scope::Repo, workspace), Scope::Repo),
    ])
}

/// Pure merge over already-resolved directories — split out from `list` so
/// the override-on-collision behavior is testable without touching the
/// real `HOME` environment variable (mutating a process-wide env var in a
/// test risks corrupting other tests running in parallel in the same
/// process).
fn merge(dirs: &[(Option<PathBuf>, Scope)]) -> Vec<AgentEntry> {
    let mut by_name: BTreeMap<String, AgentEntry> = BTreeMap::new();
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
            let Ok(agent) = serde_json::from_str::<CustomAgent>(&contents) else {
                continue;
            };
            by_name.insert(
                agent.name.clone(),
                AgentEntry {
                    agent,
                    scope: *scope,
                },
            );
        }
    }
    by_name.into_values().collect()
}

#[must_use]
pub fn load(workspace: &Path, name: &str) -> Option<AgentEntry> {
    list(workspace)
        .into_iter()
        .find(|entry| entry.agent.name == name)
}

/// # Errors
///
/// Returns an error when the target scope's directory cannot be created or
/// the file cannot be written.
pub fn save(workspace: &Path, agent: &CustomAgent, scope: Scope) -> Result<PathBuf, String> {
    let dir = dir_for(scope, workspace).ok_or("cannot locate a home directory")?;
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join(format!("{}.json", agent.name));
    let contents = serde_json::to_string_pretty(agent).map_err(|error| error.to_string())?;
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

/// Kebab-case a model-proposed name into a valid `/name` identifier:
/// lowercase, ascii-alphanumeric runs joined by single hyphens, no
/// leading/trailing hyphen.
#[must_use]
pub fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = true; // suppresses a leading hyphen
    for ch in text.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    slug.trim_end_matches('-').to_owned()
}

// ---------------------------------------------------------------------
// Conversational builder: prompts + strict-signal parsing, mirroring
// swarm.rs's BOSS_PLAN_SYSTEM/parse_verdict style.
// ---------------------------------------------------------------------

pub const AGENT_BUILDER_SYSTEM: &str = "You are helping design a new custom sub-agent for \
Junebug, a coding-agent CLI. Have a real back-and-forth with the user to understand the task \
deeply before proposing anything: what exactly the agent needs to do, what tools it genuinely \
needs (list_dir/read_file/search/semantic_search for inspection; write_file/edit_file for \
changes; run_command for builds/tests/automation; git_status/git_diff for repo state; \
web_search/fetch_url for external lookups; write_todos for multi-step planning), and whether \
it should ever be allowed to write files or run commands, or should stay strictly read-only \
regardless of the session's ambient permission — a good default for a \"verify\"/\"smoke \
test\"/\"review\" style agent that should never accidentally change anything. Ask clarifying \
questions — do not rush to a spec. Only once you are confident you understand the goal and the \
user seems satisfied, reply with exactly one ```json fenced object with fields: name (short \
kebab-case identifier, becomes /name), description (one line), system_prompt (the full \
tailored system prompt for the new agent — write this as if briefing that agent directly, in \
second person, fully self-contained: it will never see this conversation), tools (array of \
tool names from the list above), and optionally permission (one of \"read-only\", \"ask\", \
\"workspace-write\", \"yolo\" — a CAP, never wider than whatever the user is actually running \
under). End that final message with exactly one line: AGENT: READY";

#[must_use]
pub fn build_request(notes: &str) -> String {
    format!(
        "Task notes: {notes}\n\nDesign a custom agent for this. Ask whatever you need to know first."
    )
}

#[must_use]
pub fn update_request(existing: &CustomAgent, notes: &str) -> String {
    format!(
        "You are refining an EXISTING custom agent, not creating one from scratch. Current \
         definition:\nname: {}\ndescription: {}\ntools: {}\npermission: {}\nsystem_prompt:\n{}\n\n\
         Requested changes: {notes}\n\nAsk whatever you need, then reply with the FULL updated \
         spec (not a diff) in the same ```json object + AGENT: READY format.",
        existing.name,
        existing.description,
        existing.tools.join(", "),
        existing
            .permission
            .as_deref()
            .unwrap_or("(none — uses the session's ambient permission)"),
        existing.system_prompt,
    )
}

/// Strict last-non-blank-line check, mirroring `swarm::parse_verdict`'s
/// discipline: readiness is only ever trusted from this exact line, never
/// inferred from prose.
#[must_use]
pub fn parse_ready(text: &str) -> bool {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim().eq_ignore_ascii_case("AGENT: READY"))
}

#[derive(Debug, Deserialize)]
struct RawSpec {
    name: String,
    #[serde(default)]
    description: String,
    system_prompt: String,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    permission: Option<String>,
}

/// Extract the builder's proposed agent spec from its reply.
///
/// # Errors
///
/// Returns an error when no parsable spec object is present, its name
/// slugifies to nothing, or its `system_prompt` is empty.
pub fn parse_spec(text: &str) -> Result<CustomAgent, String> {
    let raw = crate::jsonblock::find_object(text)
        .ok_or("the reply contains no JSON agent spec object")?;
    let spec: RawSpec = serde_json::from_str(raw)
        .map_err(|error| format!("could not parse the agent spec: {error}"))?;
    let name = slugify(&spec.name);
    if name.is_empty() {
        return Err("the agent spec's name is empty once slugified".to_owned());
    }
    if spec.system_prompt.trim().is_empty() {
        return Err("the agent spec has an empty system_prompt".to_owned());
    }
    if let Some(permission) = &spec.permission
        && PermissionMode::parse(permission).is_err()
    {
        return Err(format!(
            "the agent spec's permission '{permission}' is not valid"
        ));
    }
    Ok(CustomAgent {
        name,
        description: if spec.description.trim().is_empty() {
            "custom agent".to_owned()
        } else {
            spec.description
        },
        system_prompt: spec.system_prompt,
        tools: spec.tools,
        permission: spec.permission,
    })
}

/// A deterministic one-line-per-agent listing for `/agents` — no model call.
#[must_use]
pub fn format_list(entries: &[AgentEntry]) -> String {
    use std::fmt::Write as _;
    if entries.is_empty() {
        return "no custom agents configured — build one with /agent-build <notes>".to_owned();
    }
    let mut out = String::new();
    for entry in entries {
        let _ = writeln!(
            out,
            "  /{:<20} [{}] {} ({} tools{})",
            entry.agent.name,
            entry.scope.label(),
            entry.agent.description,
            entry.agent.tools.len(),
            entry
                .agent
                .permission
                .as_deref()
                .map_or_else(String::new, |permission| format!(
                    ", capped at {permission}"
                )),
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
            "junebug-custom-agent-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ))
    }

    fn sample(name: &str) -> CustomAgent {
        CustomAgent {
            name: name.to_owned(),
            description: "smoke-tests the game".to_owned(),
            system_prompt: "You are a smoke-test agent.".to_owned(),
            tools: vec!["run_command".to_owned(), "read_file".to_owned()],
            permission: Some("workspace-write".to_owned()),
        }
    }

    #[test]
    fn slugify_lowercases_and_joins_with_single_hyphens() {
        assert_eq!(slugify("Game Smoke Tester!"), "game-smoke-tester");
        assert_eq!(slugify("  --weird__name--  "), "weird-name");
        assert_eq!(slugify("???"), "");
    }

    #[test]
    fn parses_a_fenced_agent_spec_and_fills_in_a_missing_description() {
        let reply = "Sounds good.\n```json\n{\"name\":\"Game Smoke Tester\",\"system_prompt\":\"You test the game.\",\"tools\":[\"run_command\"]}\n```\nAGENT: READY";
        assert!(parse_ready(reply));
        let agent = parse_spec(reply).expect("agent");
        assert_eq!(agent.name, "game-smoke-tester");
        assert_eq!(agent.description, "custom agent");
        assert_eq!(agent.tools, vec!["run_command".to_owned()]);
    }

    #[test]
    fn ready_signal_is_strict_about_trailing_prose() {
        assert!(!parse_ready("AGENT: READY\nwait, one more thing"));
        assert!(parse_ready("some text\nAGENT: READY\n"));
    }

    #[test]
    fn rejects_an_invalid_permission_name() {
        let reply =
            "```json\n{\"name\":\"a\",\"system_prompt\":\"x\",\"permission\":\"godmode\"}\n```";
        assert!(parse_spec(reply).is_err());
    }

    #[test]
    fn effective_permission_only_ever_narrows() {
        let capped = sample("a"); // capped at workspace-write
        assert_eq!(
            capped.effective_permission(PermissionMode::Yolo),
            PermissionMode::WorkspaceWrite
        );
        assert_eq!(
            capped.effective_permission(PermissionMode::ReadOnly),
            PermissionMode::ReadOnly
        );
        let uncapped = CustomAgent {
            permission: None,
            ..sample("b")
        };
        assert_eq!(
            uncapped.effective_permission(PermissionMode::Yolo),
            PermissionMode::Yolo
        );
    }

    #[test]
    fn save_load_list_and_delete_round_trip_within_repo_scope() {
        // Repo scope alone is enough to exercise save/load/list/delete
        // without touching the real HOME environment variable — see
        // `merge`'s doc comment for why that's deliberately avoided here.
        let workspace = scratch_dir("roundtrip");
        std::fs::create_dir_all(&workspace).expect("workspace");

        let agent_a = sample("game-smoke-tester");
        save(&workspace, &agent_a, Scope::Repo).expect("save a");
        let agent_b = CustomAgent {
            description: "a second agent".to_owned(),
            ..sample("second-agent")
        };
        save(&workspace, &agent_b, Scope::Repo).expect("save b");

        assert_eq!(list(&workspace).len(), 2);
        let loaded = load(&workspace, "game-smoke-tester").expect("loaded");
        assert_eq!(loaded.scope.label(), "repo");
        assert_eq!(loaded.agent.description, "smoke-tests the game");

        delete(&workspace, "second-agent", Scope::Repo).expect("delete");
        assert_eq!(list(&workspace).len(), 1);

        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn merge_lets_a_later_directory_override_an_earlier_one_on_name_collision() {
        // Exercises the actual global-vs-repo override semantics `list`
        // relies on, via the pure `merge` helper over two directories under
        // our own control — no real HOME involved.
        let global_dir = scratch_dir("merge-global");
        let repo_dir = scratch_dir("merge-repo");
        std::fs::create_dir_all(&global_dir).expect("global dir");
        std::fs::create_dir_all(&repo_dir).expect("repo dir");

        let write = |dir: &Path, agent: &CustomAgent| {
            std::fs::write(
                dir.join(format!("{}.json", agent.name)),
                serde_json::to_string_pretty(agent).expect("serialize"),
            )
            .expect("write");
        };
        write(
            &global_dir,
            &CustomAgent {
                description: "global version".to_owned(),
                ..sample("shared-name")
            },
        );
        write(
            &repo_dir,
            &CustomAgent {
                description: "repo version".to_owned(),
                ..sample("shared-name")
            },
        );

        let entries = merge(&[
            (Some(global_dir.clone()), Scope::Global),
            (Some(repo_dir.clone()), Scope::Repo),
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].agent.description, "repo version");
        assert_eq!(entries[0].scope.label(), "repo");

        std::fs::remove_dir_all(&global_dir).expect("cleanup global");
        std::fs::remove_dir_all(&repo_dir).expect("cleanup repo");
    }

    #[test]
    fn format_list_reports_when_empty() {
        assert!(format_list(&[]).contains("no custom agents configured"));
    }
}
