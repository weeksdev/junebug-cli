//! Skills: a packaged block of instructions, injected into the *current*
//! conversation's context rather than spawning a new isolated agent —
//! deliberately the lightest-weight of the three extensibility mechanisms
//! (`custom_agent`, `custom_tool`, `skill`). Mirrors the `SKILL.md`
//! convention Claude Code itself uses for skills, since that is almost
//! certainly the direct prior art anyone asking for this means.
//!
//! Stored as `~/.junebug/skills/<name>/SKILL.md` (global) and
//! `.junebug/skills/<name>/SKILL.md` (repo) — a directory per skill (room
//! for supporting files alongside the markdown later, even though v1 only
//! reads the one file), both scopes loaded and merged, repo winning a name
//! collision, the same pattern `custom_agent.rs`/`custom_tool.rs` use.
//!
//! Built conversationally via `/skill-build` under read-only tools (a
//! skill is just instructions — nothing to test the way a custom tool's
//! script needs testing), the same shape `/agent-build` already uses.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
}

pub struct SkillEntry {
    pub skill: Skill,
    pub scope: Scope,
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn dir_for(scope: Scope, workspace: &Path) -> Option<PathBuf> {
    match scope {
        Scope::Global => home().map(|home| home.join(".junebug").join("skills")),
        Scope::Repo => Some(workspace.join(".junebug").join("skills")),
    }
}

/// Split `SKILL.md`'s optional `---`-delimited frontmatter (`name: ...`,
/// `description: ...` lines) from its markdown body. Deliberately not a
/// real YAML parser — just the two fields a skill needs — falling back
/// gracefully (directory name, first body line) when frontmatter is
/// missing or a field isn't present, so a plain hand-written markdown file
/// still works as a skill.
fn parse_skill_md(directory_name: &str, contents: &str) -> Skill {
    let mut name = directory_name.to_owned();
    let mut description = String::new();
    let mut body = contents;

    if let Some(rest) = contents.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        let frontmatter = &rest[..end];
        body = rest[end + 4..].trim_start_matches('\n');
        for line in frontmatter.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().trim_matches('"').to_owned();
            match key.trim() {
                "name" if !value.is_empty() => name = value,
                "description" => description = value,
                _ => {}
            }
        }
    }

    if description.is_empty() {
        body.lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim_start_matches('#')
            .trim()
            .clone_into(&mut description);
    }

    Skill {
        name,
        description,
        body: body.trim().to_owned(),
    }
}

fn render_skill_md(skill: &Skill) -> String {
    format!(
        "---\nname: {}\ndescription: {}\n---\n\n{}\n",
        skill.name, skill.description, skill.body
    )
}

/// Every configured skill, global and repo merged — a repo skill overrides
/// a global one of the same name.
#[must_use]
pub fn list(workspace: &Path) -> Vec<SkillEntry> {
    merge(&[
        (dir_for(Scope::Global, workspace), Scope::Global),
        (dir_for(Scope::Repo, workspace), Scope::Repo),
    ])
}

/// Pure merge over already-resolved directories — see `custom_agent::merge`
/// for why this is split out (testable without touching real `HOME`).
fn merge(dirs: &[(Option<PathBuf>, Scope)]) -> Vec<SkillEntry> {
    let mut by_name: BTreeMap<String, SkillEntry> = BTreeMap::new();
    for (dir, scope) in dirs {
        let Some(dir) = dir else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(directory_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Ok(contents) = std::fs::read_to_string(path.join("SKILL.md")) else {
                continue;
            };
            let skill = parse_skill_md(directory_name, &contents);
            by_name.insert(
                skill.name.clone(),
                SkillEntry {
                    skill,
                    scope: *scope,
                },
            );
        }
    }
    by_name.into_values().collect()
}

#[must_use]
pub fn load(workspace: &Path, name: &str) -> Option<SkillEntry> {
    list(workspace)
        .into_iter()
        .find(|entry| entry.skill.name == name)
}

/// # Errors
///
/// Returns an error when the target scope's directory cannot be created or
/// the file cannot be written.
pub fn save(workspace: &Path, skill: &Skill, scope: Scope) -> Result<PathBuf, String> {
    let root = dir_for(scope, workspace).ok_or("cannot locate a home directory")?;
    let directory = root.join(&skill.name);
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join("SKILL.md");
    std::fs::write(&path, render_skill_md(skill)).map_err(|error| error.to_string())?;
    Ok(path)
}

/// # Errors
///
/// Returns an error when the skill's directory does not exist in that scope
/// or cannot be removed.
pub fn delete(workspace: &Path, name: &str, scope: Scope) -> Result<(), String> {
    let root = dir_for(scope, workspace).ok_or("cannot locate a home directory")?;
    std::fs::remove_dir_all(root.join(name)).map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------
// Conversational builder.
// ---------------------------------------------------------------------

pub const SKILL_BUILDER_SYSTEM: &str = "You are helping author a new skill for Junebug, a \
coding-agent CLI. A skill is a packaged block of instructions loaded into a conversation on \
demand — not a new agent with its own tools or identity, just knowledge/guidance the main agent \
should follow once this skill is invoked (e.g. a house style guide, a checklist for a recurring \
task, domain knowledge about this specific repo). Have a real back-and-forth first: what \
recurring task or knowledge should this capture, when should it apply, what should the agent \
concretely do differently once it has this skill loaded. You have read-only tools if you need to \
inspect the workspace to write grounded, specific instructions instead of generic advice. Once \
confident, reply with exactly one ```json fenced object with fields: name (short kebab-case \
identifier, becomes /name), description (one line — when this skill applies, used to help decide \
whether to load it), and body (the full instructions in markdown, written directly to whoever \
will follow them — second person, self-contained, concrete). End that final message with exactly \
one line: SKILL: READY";

#[must_use]
pub fn build_request(notes: &str) -> String {
    format!("Task notes: {notes}\n\nDesign a skill for this. Ask whatever you need to know first.")
}

#[must_use]
pub fn update_request(existing: &Skill, notes: &str) -> String {
    format!(
        "You are refining an EXISTING skill, not creating one from scratch. Current \
         definition:\nname: {}\ndescription: {}\nbody:\n{}\n\nRequested changes: {notes}\n\nAsk \
         whatever you need, then reply with the FULL updated spec (not a diff) in the same \
         ```json object + SKILL: READY format.",
        existing.name, existing.description, existing.body,
    )
}

/// Strict last-non-blank-line check, mirroring `swarm::parse_verdict`.
#[must_use]
pub fn parse_ready(text: &str) -> bool {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim().eq_ignore_ascii_case("SKILL: READY"))
}

#[derive(Debug, Deserialize)]
struct RawSpec {
    name: String,
    #[serde(default)]
    description: String,
    body: String,
}

/// Extract the builder's proposed skill spec from its reply.
///
/// # Errors
///
/// Returns an error when no parsable spec object is present, its name is
/// unusable, or `body` is empty.
pub fn parse_spec(text: &str) -> Result<Skill, String> {
    let raw = crate::jsonblock::find_object(text)
        .ok_or("the reply contains no JSON skill spec object")?;
    let spec: RawSpec = serde_json::from_str(raw)
        .map_err(|error| format!("could not parse the skill spec: {error}"))?;
    let name = crate::custom_agent::slugify(&spec.name);
    if name.is_empty() {
        return Err("the skill spec's name is empty once slugified".to_owned());
    }
    if spec.body.trim().is_empty() {
        return Err("the skill spec has an empty body".to_owned());
    }
    Ok(Skill {
        name,
        description: if spec.description.trim().is_empty() {
            "custom skill".to_owned()
        } else {
            spec.description
        },
        body: spec.body,
    })
}

/// A deterministic one-line-per-skill listing for `/skills` — no model call.
#[must_use]
pub fn format_list(entries: &[SkillEntry]) -> String {
    use std::fmt::Write as _;
    if entries.is_empty() {
        return "no skills configured — build one with /skill-build <notes>".to_owned();
    }
    let mut out = String::new();
    for entry in entries {
        let _ = writeln!(
            out,
            "  /{:<20} [{}] {}",
            entry.skill.name,
            entry.scope.label(),
            entry.skill.description
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
            "junebug-skill-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ))
    }

    fn sample(name: &str) -> Skill {
        Skill {
            name: name.to_owned(),
            description: "writes good changelog entries".to_owned(),
            body: "Always lead with the why, not the what.".to_owned(),
        }
    }

    #[test]
    fn parses_frontmatter_and_body() {
        let contents = "---\nname: changelog-writer\ndescription: \"writes changelogs\"\n---\n\nAlways lead with the why.\n";
        let skill = parse_skill_md("fallback-name", contents);
        assert_eq!(skill.name, "changelog-writer");
        assert_eq!(skill.description, "writes changelogs");
        assert_eq!(skill.body, "Always lead with the why.");
    }

    #[test]
    fn falls_back_to_directory_name_and_first_line_with_no_frontmatter() {
        let skill = parse_skill_md("my-skill", "# My Skill\n\nSome instructions here.");
        assert_eq!(skill.name, "my-skill");
        assert_eq!(skill.description, "My Skill");
        assert!(skill.body.contains("Some instructions here."));
    }

    #[test]
    fn save_then_load_round_trips_through_real_frontmatter() {
        let workspace = scratch_dir("roundtrip");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let skill = sample("changelog-writer");
        save(&workspace, &skill, Scope::Repo).expect("save");
        let loaded = load(&workspace, "changelog-writer").expect("loaded");
        assert_eq!(loaded.skill.description, skill.description);
        assert_eq!(loaded.skill.body, skill.body);
        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn list_and_delete_work_within_repo_scope() {
        let workspace = scratch_dir("list-delete");
        std::fs::create_dir_all(&workspace).expect("workspace");
        save(&workspace, &sample("a"), Scope::Repo).expect("save a");
        save(&workspace, &sample("b"), Scope::Repo).expect("save b");
        assert_eq!(list(&workspace).len(), 2);
        delete(&workspace, "a", Scope::Repo).expect("delete");
        assert_eq!(list(&workspace).len(), 1);
        std::fs::remove_dir_all(&workspace).expect("cleanup");
    }

    #[test]
    fn merge_lets_repo_override_global_on_name_collision() {
        let global_dir = scratch_dir("merge-global");
        let repo_dir = scratch_dir("merge-repo");
        std::fs::create_dir_all(global_dir.join("shared-name")).expect("global dir");
        std::fs::create_dir_all(repo_dir.join("shared-name")).expect("repo dir");
        std::fs::write(
            global_dir.join("shared-name").join("SKILL.md"),
            render_skill_md(&Skill {
                description: "global version".to_owned(),
                ..sample("shared-name")
            }),
        )
        .expect("write global");
        std::fs::write(
            repo_dir.join("shared-name").join("SKILL.md"),
            render_skill_md(&Skill {
                description: "repo version".to_owned(),
                ..sample("shared-name")
            }),
        )
        .expect("write repo");

        let entries = merge(&[
            (Some(global_dir.clone()), Scope::Global),
            (Some(repo_dir.clone()), Scope::Repo),
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].skill.description, "repo version");

        std::fs::remove_dir_all(&global_dir).expect("cleanup global");
        std::fs::remove_dir_all(&repo_dir).expect("cleanup repo");
    }

    #[test]
    fn parses_a_fenced_skill_spec() {
        let reply = "Here you go.\n```json\n{\"name\":\"Changelog Writer\",\"body\":\"Lead with why.\"}\n```\nSKILL: READY";
        assert!(parse_ready(reply));
        let skill = parse_spec(reply).expect("skill");
        assert_eq!(skill.name, "changelog-writer");
        assert_eq!(skill.description, "custom skill");
    }

    #[test]
    fn empty_body_is_rejected() {
        let reply = "```json\n{\"name\":\"a\",\"body\":\"\"}\n```";
        assert!(parse_spec(reply).is_err());
    }
}
