//! Model swarm: the boss/worker/checker orchestration pattern.
//!
//! One expensive "boss" model writes the spec, reviews the outcome, and rules
//! on disputes — it never writes files itself. Cheap "worker" models do every
//! task. A "checker" model independently verifies each task in the workspace
//! without trusting the worker's report; failures go back to the worker with
//! specific feedback, and repeated failures escalate to the boss for a
//! ruling. This module holds the configuration, prompts, and reply parsing;
//! the orchestration loop lives in the binary, driving `agent::run_loop`
//! once per agent turn with the role's pinned model.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A provider/model assignment for one role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmRoles {
    pub boss: Target,
    pub worker: Target,
    pub checker: Target,
}

/// Most tasks a plan may contain; larger plans are truncated.
pub const MAX_TASKS: usize = 12;
/// Worker attempts per task before the dispute escalates to the boss.
pub const MAX_ATTEMPTS: usize = 3;

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

/// The user-level swarm configuration written by `/swarm-setup`.
#[must_use]
pub fn user_config_path() -> Option<PathBuf> {
    home().map(|home| home.join(".junebug").join("swarm.json"))
}

/// Load the swarm roles: a workspace `.junebug/swarm.json` wins over the
/// user-level file. `Ok(None)` when no configuration exists yet.
///
/// # Errors
///
/// Returns an error when a configuration file exists but cannot be parsed.
pub fn load(workspace: &Path) -> Result<Option<SwarmRoles>, String> {
    for workspace_file in [
        workspace.join(".junebug").join("swarm.json"),
        workspace.join(".febo").join("swarm.json"),
    ] {
        if workspace_file.is_file() {
            return read_roles(&workspace_file).map(Some);
        }
    }
    let Some(home) = home() else {
        return Ok(None);
    };
    for path in [
        home.join(".junebug").join("swarm.json"),
        home.join(".febo").join("swarm.json"),
    ] {
        if path.is_file() {
            return read_roles(&path).map(Some);
        }
    }
    Ok(None)
}

fn read_roles(path: &Path) -> Result<SwarmRoles, String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| format!("{}: {error}", path.display()))
}

/// Save roles to the user-level configuration and return the path written.
///
/// # Errors
///
/// Returns an error when the home directory is unknown or the file cannot
/// be written.
pub fn save(roles: &SwarmRoles) -> Result<PathBuf, String> {
    let path = user_config_path().ok_or("cannot locate a home directory")?;
    save_to(roles, &path)?;
    Ok(path)
}

/// # Errors
///
/// Returns an error when the file cannot be written.
pub fn save_to(roles: &SwarmRoles, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let contents = serde_json::to_string_pretty(roles).map_err(|error| error.to_string())?;
    fs::write(path, contents).map_err(|error| error.to_string())
}

/// One unit of work in the boss's plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    #[serde(default)]
    pub id: usize,
    pub title: String,
    pub instructions: String,
    pub check: String,
}

/// Extract the JSON task array from the boss's plan reply.
///
/// # Errors
///
/// Returns an error when no parsable task array is present or it is empty.
pub fn parse_tasks(text: &str) -> Result<Vec<Task>, String> {
    let start = match text.find("```json") {
        Some(fence) => text[fence..].find('[').map(|offset| fence + offset),
        None => text.find('['),
    }
    .ok_or("the plan contains no JSON task array")?;
    let end = text
        .rfind(']')
        .ok_or("the plan contains no JSON task array")?;
    if start > end {
        return Err("the plan contains no JSON task array".to_owned());
    }
    let mut tasks: Vec<Task> = serde_json::from_str(&text[start..=end])
        .map_err(|error| format!("could not parse the task array: {error}"))?;
    if tasks.is_empty() {
        return Err("the plan contains no tasks".to_owned());
    }
    tasks.truncate(MAX_TASKS);
    for (index, task) in tasks.iter_mut().enumerate() {
        task.id = index + 1;
    }
    Ok(tasks)
}

/// On-disk progress of a swarm run, written after planning and after every
/// finished task so an aborted swarm can resume where it left off with
/// `/swarm resume`. Cleared when the swarm completes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmState {
    pub goal: String,
    pub constitution: String,
    pub tasks: Vec<Task>,
    /// `(task id, "done" | "FAILED")` for finished tasks; ids not listed are
    /// still pending.
    pub outcomes: Vec<(usize, String)>,
    pub reworks: usize,
    pub failures: usize,
}

impl SwarmState {
    #[must_use]
    pub fn is_finished(&self, task_id: usize) -> bool {
        self.outcomes.iter().any(|(id, _)| *id == task_id)
    }
}

#[must_use]
pub fn state_path(workspace: &Path) -> PathBuf {
    workspace.join(".junebug").join("swarm_state.json")
}

/// Load the saved progress of an aborted swarm, if any.
///
/// # Errors
///
/// Returns an error when a state file exists but cannot be parsed.
pub fn load_state(workspace: &Path) -> Result<Option<SwarmState>, String> {
    let path = state_path(workspace);
    if !path.is_file() {
        return Ok(None);
    }
    let contents = fs::read_to_string(&path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents)
        .map(Some)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// # Errors
///
/// Returns an error when the state file cannot be written.
pub fn save_state(workspace: &Path, state: &SwarmState) -> Result<(), String> {
    let path = state_path(workspace);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let contents = serde_json::to_string_pretty(state).map_err(|error| error.to_string())?;
    fs::write(path, contents).map_err(|error| error.to_string())
}

/// Remove the state file after a completed swarm. Best effort.
pub fn clear_state(workspace: &Path) {
    let _ = fs::remove_file(state_path(workspace));
}

/// True when a provider error looks like a transient stream or network
/// hiccup (e.g. reqwest's "request or response body error" when a stream
/// dies mid-turn) rather than a configuration problem, so the agent turn is
/// worth retrying from scratch instead of aborting the whole swarm.
#[must_use]
pub fn is_transient_provider_error(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    [
        "body error",
        "connection",
        "timed out",
        "timeout",
        "reset",
        "broken pipe",
        "unexpected eof",
        "temporarily",
        "overloaded",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "incomplete message",
    ]
    .iter()
    .any(|needle| error.contains(needle))
}

/// The constitution portion of the plan: everything before the task array.
#[must_use]
pub fn constitution_of(plan: &str) -> String {
    let cut = plan
        .find("```json")
        .or_else(|| plan.find('['))
        .unwrap_or(plan.len());
    let constitution = plan[..cut].trim();
    if constitution.is_empty() {
        "Do exactly what each task specifies; nothing more.".to_owned()
    } else {
        constitution.to_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail(String),
}

/// Parse the checker's mandatory `VERDICT:` line (last one wins). A missing
/// verdict is a failure: silence must never count as a pass.
#[must_use]
pub fn parse_verdict(text: &str) -> Verdict {
    for line in text.lines().rev() {
        let Some(rest) = line.trim().strip_prefix("VERDICT:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.to_ascii_uppercase().starts_with("PASS") {
            return Verdict::Pass;
        }
        let reason = rest
            .trim_start_matches(|c: char| c.is_ascii_alphabetic())
            .trim_start_matches(':')
            .trim();
        return Verdict::Fail(if reason.is_empty() {
            "unspecified failure".to_owned()
        } else {
            reason.to_owned()
        });
    }
    Verdict::Fail("the checker did not return a VERDICT line".to_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ruling {
    /// The work is acceptable; the checker is overruled.
    Worker,
    /// The checker was right; the worker gets one final guided rework.
    Checker,
}

/// Parse the boss's mandatory `RULING:` line. Defaults to upholding the
/// checker: an unclear ruling must never accept unverified work.
#[must_use]
pub fn parse_ruling(text: &str) -> Ruling {
    for line in text.lines().rev() {
        let upper = line.trim().to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("RULING:") {
            return if rest.trim().starts_with("WORKER") {
                Ruling::Worker
            } else {
                Ruling::Checker
            };
        }
    }
    Ruling::Checker
}

pub const BOSS_PLAN_SYSTEM: &str = "You are the boss agent of a model swarm. You never write files or code yourself — cheaper worker models do that. Inspect the workspace with the read-only tools first, then reply with exactly two things:\n1. CONSTITUTION: a short numbered list of standards that define what done-right means for this goal.\n2. A ```json fenced code block containing an array of 3 to 12 tasks, each an object with exactly these string fields: \"title\", \"instructions\", \"check\". instructions must be fully self-contained for a worker that sees nothing else but the constitution. check must describe concrete verification that a separate checking agent can execute with read/search/command tools, without trusting the worker's report.";

#[must_use]
pub fn plan_request(goal: &str) -> String {
    format!(
        "Goal: {goal}\n\nInspect the workspace, then produce the constitution and the JSON task array."
    )
}

pub const WORKER_SYSTEM: &str = "You are a worker agent in a model swarm. Complete exactly the assigned task — nothing more, nothing less. Follow the constitution. Do real work with tools; never claim something is done unless a tool result proves it. Finish with a one-paragraph report of what you did.";

#[must_use]
pub fn worker_request(constitution: &str, task: &Task, feedback: Option<&str>) -> String {
    let mut request = format!(
        "CONSTITUTION:\n{constitution}\n\nTASK {}: {}\n{}",
        task.id, task.title, task.instructions
    );
    if let Some(feedback) = feedback {
        let _ = write!(
            request,
            "\n\nA checking agent rejected the previous attempt. Fix precisely this, and nothing else:\n{feedback}"
        );
    }
    request
}

pub const CHECKER_SYSTEM: &str = "You are a checking agent in a model swarm. Independently verify that the task below is actually complete in the workspace. Never trust the worker's report — verify everything yourself with read, search, and command tools. Be strict but honest: enforce the CHECK criteria, not preferences of your own. Your reply MUST end with exactly one line, either:\nVERDICT: PASS\nVERDICT: FAIL: <specific, actionable reasons>";

#[must_use]
pub fn checker_request(task: &Task) -> String {
    format!(
        "TASK {}: {}\n{}\n\nCHECK:\n{}",
        task.id, task.title, task.instructions, task.check
    )
}

pub const BOSS_RULING_SYSTEM: &str = "You are the boss agent of a model swarm, ruling on a dispute after a task repeatedly failed its checks. Inspect the workspace yourself where needed. Decide whether the work truly satisfies the task and the constitution — checkers can be wrong too, and honesty beats padding. Your reply MUST end with exactly one line, either:\nRULING: WORKER (the work is acceptable; the checker is overruled)\nRULING: CHECKER: <one short paragraph of precise guidance for the final rework>";

#[must_use]
pub fn ruling_request(
    constitution: &str,
    task: &Task,
    worker_report: &str,
    checker_feedback: &str,
) -> String {
    format!(
        "CONSTITUTION:\n{constitution}\n\nTASK {}: {}\n{}\n\nCHECK:\n{}\n\nWORKER'S REPORT:\n{}\n\nCHECKER'S LAST FEEDBACK:\n{}",
        task.id,
        task.title,
        task.instructions,
        task.check,
        if worker_report.is_empty() {
            "(none)"
        } else {
            worker_report
        },
        checker_feedback
    )
}

pub const BOSS_REVIEW_SYSTEM: &str = "You are the boss agent of a model swarm. The build is finished. Review the outcome against the constitution and write a short final report for the human: what shipped, what was reworked and why, and anything that still needs human attention. Do not sugarcoat failures.";

#[must_use]
pub fn review_request(goal: &str, outcomes: &str, diff: &str) -> String {
    let mut request = format!("Goal: {goal}\n\nTask outcomes:\n{outcomes}");
    if !diff.trim().is_empty() {
        let _ = write!(request, "\n\nWorkspace diff:\n{diff}");
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tasks_from_a_fenced_plan_and_renumbers() {
        let plan = "CONSTITUTION:\n1. Be honest.\n\n```json\n[{\"id\":9,\"title\":\"a\",\"instructions\":\"do a\",\"check\":\"verify a\"},{\"title\":\"b\",\"instructions\":\"do b\",\"check\":\"verify b\"}]\n```\nDone.";
        let tasks = parse_tasks(plan).expect("tasks");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].id, 1);
        assert_eq!(tasks[1].id, 2);
        assert_eq!(tasks[1].title, "b");
        assert_eq!(constitution_of(plan), "CONSTITUTION:\n1. Be honest.");
    }

    #[test]
    fn plan_without_tasks_is_an_error() {
        assert!(parse_tasks("no json here").is_err());
        assert!(parse_tasks("```json\n[]\n```").is_err());
    }

    #[test]
    fn verdict_parsing_is_strict_about_silence() {
        assert_eq!(parse_verdict("all good\nVERDICT: PASS"), Verdict::Pass);
        assert_eq!(
            parse_verdict("VERDICT: FAIL: missing alt text"),
            Verdict::Fail("missing alt text".to_owned())
        );
        assert!(matches!(
            parse_verdict("looks fine to me!"),
            Verdict::Fail(reason) if reason.contains("did not return")
        ));
        // The last verdict line wins.
        assert_eq!(
            parse_verdict("VERDICT: PASS\nwait, no\nVERDICT: FAIL: broken link"),
            Verdict::Fail("broken link".to_owned())
        );
    }

    #[test]
    fn ruling_defaults_to_the_checker() {
        assert_eq!(parse_ruling("RULING: WORKER"), Ruling::Worker);
        assert_eq!(
            parse_ruling("RULING: CHECKER: tighten the check"),
            Ruling::Checker
        );
        assert_eq!(parse_ruling("hmm, unclear"), Ruling::Checker);
    }

    #[test]
    fn swarm_state_round_trips_and_clears() {
        let root = std::env::temp_dir().join(format!(
            "junebug-swarm-state-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        assert_eq!(load_state(&root).expect("empty"), None);
        let state = SwarmState {
            goal: "build the thing".to_owned(),
            constitution: "1. honesty".to_owned(),
            tasks: vec![Task {
                id: 1,
                title: "a".to_owned(),
                instructions: "do a".to_owned(),
                check: "verify a".to_owned(),
            }],
            outcomes: vec![(1, "done".to_owned())],
            reworks: 2,
            failures: 0,
        };
        save_state(&root, &state).expect("save");
        let loaded = load_state(&root).expect("load").expect("present");
        assert_eq!(loaded, state);
        assert!(loaded.is_finished(1));
        assert!(!loaded.is_finished(2));
        clear_state(&root);
        assert_eq!(load_state(&root).expect("cleared"), None);
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn transient_provider_errors_are_recognized() {
        assert!(is_transient_provider_error(
            "request or response body error"
        ));
        assert!(is_transient_provider_error("Connection reset by peer"));
        assert!(is_transient_provider_error("HTTP 503 Service Unavailable"));
        assert!(!is_transient_provider_error("invalid API key"));
        assert!(!is_transient_provider_error("model not found: deepseek-v9"));
    }

    #[test]
    fn roles_round_trip_through_the_config_file() {
        let path = std::env::temp_dir()
            .join(format!(
                "junebug-swarm-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ))
            .join("swarm.json");
        let roles = SwarmRoles {
            boss: Target {
                provider: "openai".to_owned(),
                model: "gpt-5.6".to_owned(),
            },
            worker: Target {
                provider: "deepseek".to_owned(),
                model: "deepseek-chat".to_owned(),
            },
            checker: Target {
                provider: "deepseek".to_owned(),
                model: "deepseek-v4-flash".to_owned(),
            },
        };
        save_to(&roles, &path).expect("save");
        assert_eq!(read_roles(&path).expect("read"), roles);
        std::fs::remove_dir_all(path.parent().expect("parent")).expect("cleanup");
    }

    #[test]
    fn loads_legacy_workspace_swarm_config() {
        let root = std::env::temp_dir().join(format!(
            "junebug-legacy-swarm-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let path = root.join(".febo/swarm.json");
        let roles = SwarmRoles {
            boss: Target {
                provider: "openai".to_owned(),
                model: "boss".to_owned(),
            },
            worker: Target {
                provider: "deepseek".to_owned(),
                model: "worker".to_owned(),
            },
            checker: Target {
                provider: "anthropic".to_owned(),
                model: "checker".to_owned(),
            },
        };
        save_to(&roles, &path).expect("legacy save");
        assert_eq!(load(&root).expect("load"), Some(roles));
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
