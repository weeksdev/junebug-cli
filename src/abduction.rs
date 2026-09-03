//! Abductive/Bayesian reasoning harness: `/investigate`.
//!
//! Where `/swarm` prevents a single model from grading its own code, this
//! module prevents a single model pass from grading its own *reasoning*.
//! Four roles run in sequence: a Generator proposes several materially
//! different explanations for a set of observations, an Evaluator scores
//! each one *blind* to its siblings (a fresh call per hypothesis, seeing
//! only that hypothesis), a Skeptic red-teams the ranked result, and a
//! Synthesizer writes the final answer under a fixed FACT/INFERENCE/
//! UNCERTAINTY/ALTERNATIVES/DISCRIMINATOR structure. The actual posterior
//! arithmetic never runs inside a prompt — see `bayes`.
//!
//! The orchestration loop lives in the binary (`investigate_agent`/
//! `run_investigation` in `main.rs`), driving `agent::run_loop` once per
//! role call with a fresh message list, exactly like `swarm_agent` does for
//! `/swarm`. This module holds the data model, persistence, reply parsing,
//! and prompts.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::swarm::Target;

/// Most hypotheses a Generator reply may contain; larger sets are truncated.
pub const MAX_HYPOTHESES: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Generating,
    Evaluating,
    Reviewing,
    Done,
    Failed,
}

/// One estimated data point behind a hypothesis's posterior. `given_h` and
/// `given_not_h` are the model's `P(E|H)`/`P(E|¬H)` estimates — never
/// trusted arithmetic, only ever fed into `bayes::likelihood_ratio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub observation: String,
    pub given_h: f64,
    pub given_not_h: f64,
    /// Derived mechanically from the ratio ("supports" / "contradicts" /
    /// "neutral") at parse time — display sugar, not a separate signal.
    pub stance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub claim: String,
    /// One of the Generator's forced diversity categories — "mundane",
    /// "systemic", "intentional", "measurement-error", "contrarian" — or a
    /// free-text label if the model adds another. Not enforced by an enum:
    /// the prompt requires the categories, the parser doesn't.
    pub category: String,
    pub prior: f64,
    pub evidence: Vec<Evidence>,
    /// Set by `bayes::posterior` + `bayes::normalize` once evaluation
    /// finishes. Zero until then.
    pub posterior: f64,
    pub assumptions: Vec<String>,
    pub falsifiers: Vec<String>,
    /// Filled in by the Skeptic pass; empty until then.
    pub critique: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Investigation {
    pub id: String,
    pub question: String,
    pub observations: Vec<String>,
    pub hypotheses: Vec<Hypothesis>,
    pub status: Status,
    /// The Synthesizer's final write-up. `None` until the Reviewing phase
    /// finishes.
    pub synthesis: Option<String>,
    pub session_log: PathBuf,
    pub created_at: u64,
    pub updated_at: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------
// Persistence: one JSON file per investigation, mirroring swarm.rs's
// state_path/load_state/save_state but one-per-run instead of
// one-per-workspace.
// ---------------------------------------------------------------------

#[must_use]
pub fn dir(workspace: &Path) -> PathBuf {
    workspace.join(".junebug").join("investigations")
}

/// Kebab-case title (first six words) plus a short hex suffix from the
/// current time, so two investigations with the same question don't
/// collide. Collisions only need to be unlikely, not impossible.
#[must_use]
pub fn slug_for(question: &str) -> String {
    let kebab: String = question
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed: String = kebab
        .split('-')
        .filter(|s| !s.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    if trimmed.is_empty() {
        format!("investigation-{nanos:x}")
    } else {
        format!("{trimmed}-{nanos:x}")
    }
}

/// # Errors
///
/// Returns an error when the investigation cannot be serialized or written.
pub fn save(workspace: &Path, investigation: &mut Investigation) -> Result<(), String> {
    investigation.updated_at = now_secs();
    let path = dir(workspace).join(format!("{}.json", investigation.id));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let contents =
        serde_json::to_string_pretty(investigation).map_err(|error| error.to_string())?;
    std::fs::write(path, contents).map_err(|error| error.to_string())
}

/// # Errors
///
/// Returns an error when no investigation named `id` exists here, or it
/// cannot be parsed.
pub fn load(workspace: &Path, id: &str) -> Result<Investigation, String> {
    let path = dir(workspace).join(format!("{id}.json"));
    let contents =
        std::fs::read_to_string(&path).map_err(|_| format!("no investigation named '{id}'"))?;
    serde_json::from_str(&contents).map_err(|error| format!("{}: {error}", path.display()))
}

/// Every investigation saved in this workspace, newest `updated_at` first.
///
/// # Errors
///
/// Returns an error only when the directory exists but cannot be read.
pub fn list(workspace: &Path) -> Result<Vec<Investigation>, String> {
    let path = dir(workspace);
    if !path.is_dir() {
        return Ok(Vec::new());
    }
    let mut items = Vec::new();
    for entry in std::fs::read_dir(&path).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(entry.path())
            && let Ok(investigation) = serde_json::from_str::<Investigation>(&contents)
        {
            items.push(investigation);
        }
    }
    items.sort_by_key(|item| std::cmp::Reverse(item.updated_at));
    Ok(items)
}

/// The most recently updated investigation still in progress (any status
/// other than `Done`/`Failed`), if any — used by `/investigate-status` with
/// no id given.
#[must_use]
pub fn latest(workspace: &Path) -> Option<Investigation> {
    list(workspace).ok()?.into_iter().next()
}

// ---------------------------------------------------------------------
// Optional role assignment: `.junebug/investigate.json`, mirroring
// swarm.rs's `.junebug/swarm.json`. Unlike `/swarm`, this is optional —
// `/investigate` works with zero setup by defaulting every role to the
// REPL's currently active provider/model.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestigateRoles {
    pub generator: Target,
    pub evaluator: Target,
    pub skeptic: Target,
    pub synthesizer: Target,
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

/// The user-level configuration written by `/investigate-setup`.
#[must_use]
pub fn user_config_path() -> Option<PathBuf> {
    home().map(|home| home.join(".junebug").join("investigate.json"))
}

/// Load role assignments: a workspace `.junebug/investigate.json` wins over
/// the user-level file. `Ok(None)` when neither exists.
///
/// # Errors
///
/// Returns an error when a configuration file exists but cannot be parsed.
pub fn load_roles(workspace: &Path) -> Result<Option<InvestigateRoles>, String> {
    let workspace_file = workspace.join(".junebug").join("investigate.json");
    if workspace_file.is_file() {
        return read_roles(&workspace_file).map(Some);
    }
    let Some(path) = user_config_path() else {
        return Ok(None);
    };
    if path.is_file() {
        return read_roles(&path).map(Some);
    }
    Ok(None)
}

fn read_roles(path: &Path) -> Result<InvestigateRoles, String> {
    let contents = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| format!("{}: {error}", path.display()))
}

/// Save role assignments to the user-level configuration and return the
/// path written.
///
/// # Errors
///
/// Returns an error when the home directory is unknown or the file cannot
/// be written.
pub fn save_roles(roles: &InvestigateRoles) -> Result<PathBuf, String> {
    let path = user_config_path().ok_or("cannot locate a home directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let contents = serde_json::to_string_pretty(roles).map_err(|error| error.to_string())?;
    std::fs::write(&path, contents).map_err(|error| error.to_string())?;
    Ok(path)
}

// ---------------------------------------------------------------------
// Reply parsing.
// ---------------------------------------------------------------------

/// Some models reply with a bare JSON number for `id` (e.g. `"id": 1`)
/// despite the prompt asking for a string — observed live with a local
/// Ollama model. Accept either and stringify, rather than failing the whole
/// hypothesis array over one field's type.
fn flexible_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Value::deserialize(deserializer)? {
        Value::String(text) => text,
        Value::Number(number) => number.to_string(),
        _ => String::new(),
    })
}

#[derive(Debug, Deserialize)]
struct RawHypothesis {
    #[serde(default, deserialize_with = "flexible_id")]
    id: String,
    claim: String,
    #[serde(default)]
    category: String,
    #[serde(default = "default_prior")]
    prior: f64,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    falsifiers: Vec<String>,
}

const fn default_prior() -> f64 {
    0.2
}

/// Extract the Generator's JSON hypothesis array from its reply.
///
/// # Errors
///
/// Returns an error when no parsable, non-empty hypothesis array is present.
pub fn parse_hypotheses(text: &str) -> Result<Vec<Hypothesis>, String> {
    let raw =
        crate::jsonblock::find_array(text).ok_or("the reply contains no JSON hypothesis array")?;
    let mut items: Vec<RawHypothesis> = serde_json::from_str(raw)
        .map_err(|error| format!("could not parse the hypothesis array: {error}"))?;
    if items.is_empty() {
        return Err("the reply contains no hypotheses".to_owned());
    }
    items.truncate(MAX_HYPOTHESES);
    Ok(items
        .into_iter()
        .enumerate()
        .map(|(index, item)| Hypothesis {
            id: if item.id.is_empty() {
                format!("H{}", index + 1)
            } else {
                item.id
            },
            claim: item.claim,
            category: if item.category.is_empty() {
                "uncategorized".to_owned()
            } else {
                item.category
            },
            prior: item.prior.clamp(0.01, 0.99),
            evidence: Vec::new(),
            posterior: 0.0,
            assumptions: item.assumptions,
            falsifiers: item.falsifiers,
            critique: Vec::new(),
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct RawEvidence {
    observation: String,
    given_h: f64,
    given_not_h: f64,
}

fn stance_for(ratio: f64) -> &'static str {
    if ratio > 1.1 {
        "supports"
    } else if ratio < 0.9 {
        "contradicts"
    } else {
        "neutral"
    }
}

/// Extract an Evaluator's JSON evidence array from its reply.
///
/// # Errors
///
/// Returns an error when no parsable, non-empty evidence array is present.
pub fn parse_evidence(text: &str) -> Result<Vec<Evidence>, String> {
    let raw =
        crate::jsonblock::find_array(text).ok_or("the reply contains no JSON evidence array")?;
    let items: Vec<RawEvidence> = serde_json::from_str(raw)
        .map_err(|error| format!("could not parse the evidence array: {error}"))?;
    if items.is_empty() {
        return Err("the reply contains no evidence".to_owned());
    }
    Ok(items
        .into_iter()
        .map(|item| {
            let ratio = crate::bayes::likelihood_ratio(item.given_h, item.given_not_h);
            Evidence {
                observation: item.observation,
                given_h: item.given_h,
                given_not_h: item.given_not_h,
                stance: stance_for(ratio).to_owned(),
            }
        })
        .collect())
}

// ---------------------------------------------------------------------
// Prompts.
// ---------------------------------------------------------------------

pub const GENERATOR_SYSTEM: &str = "You are the hypothesis-generation agent of an \
abductive reasoning harness. Given a question and observations, generate at least \
5 materially different causal explanations — do not converge on one early. \
Include at least: one mundane/ordinary explanation, one institutional or systemic \
explanation, one explanation involving intentional action, one explanation based \
on measurement or reporting error, and one explanation that contradicts the \
apparent narrative. You may use the read-only tools available to you to inspect \
the workspace first if that would help. For each hypothesis give: a short claim, \
a category label (mundane, systemic, intentional, measurement-error, contrarian, \
or another short label if none fit), a rough prior in (0,1) that is your genuine \
best guess rather than a hedge toward 0.5, hidden assumptions, and falsifiers \
(what would prove this wrong). Reply with a ```json fenced array of objects with \
fields: id, claim, category, prior, assumptions (array of strings), falsifiers \
(array of strings). Nothing after the closing fence.";

#[must_use]
pub fn generate_request(question: &str, observations: &[String]) -> String {
    let mut request = format!("Question: {question}");
    if !observations.is_empty() {
        let _ = write!(request, "\n\nObservations:");
        for observation in observations {
            let _ = write!(request, "\n- {observation}");
        }
    }
    request
}

pub const EVALUATOR_SYSTEM: &str = "You are an evidence-evaluation agent in an \
abductive reasoning harness. You are shown exactly one hypothesis and a list of \
observations — you do not know what other hypotheses exist, and must not \
speculate about them or hedge toward a middle estimate because of that. For each \
observation, estimate P(observation | this hypothesis is true) and P(observation \
| this hypothesis is false), both in (0,1]. Be honest: most observations are only \
weak evidence one way or the other — reserve extreme ratios for observations that \
genuinely could barely occur, or could only occur, under this hypothesis. Reply \
with a ```json fenced array of objects with fields: observation, given_h, \
given_not_h. Nothing after the closing fence.";

#[must_use]
pub fn evaluate_request(hypothesis: &Hypothesis, observations: &[String]) -> String {
    let mut request = format!("Hypothesis: {}", hypothesis.claim);
    if !observations.is_empty() {
        let _ = write!(request, "\n\nObservations:");
        for observation in observations {
            let _ = write!(request, "\n- {observation}");
        }
    }
    request
}

pub const SKEPTIC_SYSTEM: &str = "You are the adversarial red-team agent of an \
abductive reasoning harness. You are shown the full ranked hypothesis list with \
its evidence. Your only job is to attack the leading hypothesis: name evidence \
that does not actually support it as strongly as claimed, alternative \
explanations for each supporting observation, base-rate neglect, selection bias, \
confirmation bias, hidden assumptions, omitted variables, and causal-direction \
errors. Do not be diplomatic. If the leading hypothesis genuinely holds up after \
real scrutiny, say so plainly and briefly instead of manufacturing criticism — \
false balance is its own failure mode.";

#[must_use]
pub fn critique_request(ranked_summary: &str) -> String {
    format!("Ranked hypotheses with evidence:\n\n{ranked_summary}")
}

pub const SYNTHESIZER_SYSTEM: &str = "You are the final synthesis agent of an \
abductive reasoning harness. Given the ranked hypotheses, their evidence, and the \
red-team critique, write the answer for a human reader under exactly these \
headings: FACT (directly established, not inferred), INFERENCE (the \
best-supported explanation and why), UNCERTAINTY (what remains genuinely \
unknown), ALTERNATIVES (other hypotheses that are not ruled out, with why they \
rank lower), DISCRIMINATOR (what evidence would most change this conclusion if \
found). Do not present the numeric posteriors as more precise than they are — \
describe confidence qualitatively (e.g. \"clearly favored\", \"a close call \
between H1 and H3\"), not as a bare percentage.";

#[must_use]
pub fn synthesize_request(ranked_summary: &str, critique: &str) -> String {
    format!("Ranked hypotheses:\n\n{ranked_summary}\n\nRed-team critique:\n\n{critique}")
}

/// A deterministic progress/result readout built purely from the saved
/// investigation — no model call. Used by `/investigate-status` and (later)
/// a browser detail pane.
#[must_use]
pub fn format_summary(investigation: &Investigation) -> String {
    let mut ranked = investigation.hypotheses.clone();
    ranked.sort_by(|a, b| {
        b.posterior
            .partial_cmp(&a.posterior)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = String::new();
    let _ = writeln!(out, "investigate — {}", investigation.question);
    let _ = writeln!(out, "status: {:?}", investigation.status);
    for hypothesis in &ranked {
        let _ = writeln!(
            out,
            "  {:>5.1}%  [{}] {}",
            hypothesis.posterior * 100.0,
            hypothesis.category,
            hypothesis.claim
        );
        if !hypothesis.critique.is_empty() {
            for note in &hypothesis.critique {
                let _ = writeln!(out, "           ⚠ {note}");
            }
        }
    }
    if let Some(synthesis) = &investigation.synthesis {
        let _ = writeln!(out, "\n{synthesis}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_investigation() -> Investigation {
        Investigation {
            id: "why-slow-abc123".to_owned(),
            question: "why did the build get slower".to_owned(),
            observations: vec!["CI went from 4m to 11m".to_owned()],
            hypotheses: vec![
                Hypothesis {
                    id: "H1".to_owned(),
                    claim: "a dependency bump added a slow codegen step".to_owned(),
                    category: "mundane".to_owned(),
                    prior: 0.3,
                    evidence: Vec::new(),
                    posterior: 0.7,
                    assumptions: vec![],
                    falsifiers: vec![],
                    critique: vec![],
                },
                Hypothesis {
                    id: "H2".to_owned(),
                    claim: "the CI runner class was silently downgraded".to_owned(),
                    category: "systemic".to_owned(),
                    prior: 0.2,
                    evidence: Vec::new(),
                    posterior: 0.3,
                    assumptions: vec![],
                    falsifiers: vec![],
                    critique: vec![],
                },
            ],
            status: Status::Done,
            synthesis: Some("FACT: ...".to_owned()),
            session_log: PathBuf::from("/tmp/session.jsonl"),
            created_at: 1,
            updated_at: 2,
        }
    }

    #[test]
    fn investigation_round_trips_through_save_and_load() {
        let root = std::env::temp_dir().join(format!(
            "junebug-investigation-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let mut investigation = sample_investigation();
        save(&root, &mut investigation).expect("save");
        let loaded = load(&root, &investigation.id).expect("load");
        assert_eq!(loaded.question, investigation.question);
        assert_eq!(loaded.hypotheses.len(), 2);
        assert!(loaded.updated_at > 0);
        let all = list(&root).expect("list");
        assert_eq!(all.len(), 1);
        assert_eq!(latest(&root).expect("latest").id, investigation.id);
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn list_on_a_missing_directory_is_empty_not_an_error() {
        let root = std::env::temp_dir().join(format!(
            "junebug-investigation-missing-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        assert!(list(&root).expect("empty").is_empty());
    }

    #[test]
    fn slug_is_kebab_case_and_capped_at_six_words() {
        let slug = slug_for("Why Did The Nightly Build Suddenly Get So Much Slower?");
        assert!(slug.starts_with("why-did-the-nightly-build-suddenly-"));
        assert!(slug.split('-').count() <= 8); // 6 words + hex suffix segment(s)
    }

    #[test]
    fn slug_falls_back_when_the_question_has_no_alphanumerics() {
        let slug = slug_for("???");
        assert!(slug.starts_with("investigation-"));
    }

    #[test]
    fn parses_hypotheses_from_a_fenced_generator_reply() {
        let reply = "Here are five explanations:\n```json\n[\
            {\"id\":\"H1\",\"claim\":\"a\",\"category\":\"mundane\",\"prior\":0.4,\"assumptions\":[],\"falsifiers\":[]},\
            {\"claim\":\"b\",\"category\":\"systemic\",\"prior\":0.2,\"assumptions\":[\"x\"],\"falsifiers\":[\"y\"]}\
            ]\n```\nDone.";
        let hypotheses = parse_hypotheses(reply).expect("hypotheses");
        assert_eq!(hypotheses.len(), 2);
        assert_eq!(hypotheses[0].id, "H1");
        assert_eq!(hypotheses[1].id, "H2"); // missing id is filled in
        assert_eq!(hypotheses[1].assumptions, vec!["x".to_owned()]);
    }

    #[test]
    fn a_bare_numeric_id_is_accepted_and_stringified() {
        // Observed live with a local Ollama model (qwen3:8b): it replied
        // with `"id": 1` (a JSON number) despite the prompt asking for a
        // string. This must not fail the whole hypothesis array.
        let reply =
            "```json\n[{\"id\":1,\"claim\":\"a\",\"category\":\"mundane\",\"prior\":0.3}]\n```";
        let hypotheses = parse_hypotheses(reply).expect("hypotheses");
        assert_eq!(hypotheses[0].id, "1");
    }

    #[test]
    fn empty_hypothesis_array_is_an_error() {
        assert!(parse_hypotheses("```json\n[]\n```").is_err());
        assert!(parse_hypotheses("no json here").is_err());
    }

    #[test]
    fn hypothesis_array_longer_than_max_is_truncated() {
        let items: Vec<String> = (0..12)
            .map(|i| format!("{{\"claim\":\"h{i}\",\"category\":\"x\",\"prior\":0.1}}"))
            .collect();
        let reply = format!("```json\n[{}]\n```", items.join(","));
        let hypotheses = parse_hypotheses(&reply).expect("hypotheses");
        assert_eq!(hypotheses.len(), MAX_HYPOTHESES);
    }

    #[test]
    fn parses_evidence_and_derives_stance() {
        let reply = "```json\n[\
            {\"observation\":\"a\",\"given_h\":0.9,\"given_not_h\":0.1},\
            {\"observation\":\"b\",\"given_h\":0.1,\"given_not_h\":0.9},\
            {\"observation\":\"c\",\"given_h\":0.5,\"given_not_h\":0.5}\
            ]\n```";
        let evidence = parse_evidence(reply).expect("evidence");
        assert_eq!(evidence.len(), 3);
        assert_eq!(evidence[0].stance, "supports");
        assert_eq!(evidence[1].stance, "contradicts");
        assert_eq!(evidence[2].stance, "neutral");
    }

    #[test]
    fn empty_evidence_array_is_an_error() {
        assert!(parse_evidence("```json\n[]\n```").is_err());
    }

    #[test]
    fn format_summary_ranks_by_posterior_and_includes_synthesis() {
        let investigation = sample_investigation();
        let summary = format_summary(&investigation);
        assert!(summary.contains("investigate — why did the build get slower"));
        let h1_pos = summary.find("70.0%").expect("h1 shown");
        let h2_pos = summary.find("30.0%").expect("h2 shown");
        assert!(h1_pos < h2_pos, "higher posterior should rank first");
        assert!(summary.contains("FACT: ..."));
    }

    #[test]
    fn roles_round_trip_through_the_config_file() {
        let target = |model: &str| Target {
            provider: "openai".to_owned(),
            model: model.to_owned(),
        };
        let roles = InvestigateRoles {
            generator: target("gpt-5.6"),
            evaluator: target("gpt-5.6-mini"),
            skeptic: target("gpt-5.6-mini"),
            synthesizer: target("gpt-5.6"),
        };
        let path = std::env::temp_dir()
            .join(format!(
                "junebug-investigate-roles-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ))
            .join("investigate.json");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dir");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&roles).expect("serialize"),
        )
        .expect("write");
        assert_eq!(read_roles(&path).expect("read"), roles);
        std::fs::remove_dir_all(path.parent().expect("parent")).expect("cleanup");
    }
}
