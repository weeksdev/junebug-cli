# `/investigate` mode — implementation spec for handoff

## Status: shipped (v1, all phases below)

Implemented, verified, and live-tested end-to-end. Summary of what landed,
in build order:

1. **Shared JSON-block scanner** — `src/jsonblock.rs` (new), `swarm::parse_tasks`
   refactored to use it. Pure refactor, `/swarm`'s existing tests confirm no
   behavior change.
2. **`src/bayes.rs`** — as specified in §3, unit-tested (14 tests: likelihood
   ratio direction, sequential compounding, unit-interval clamping,
   normalization, entropy, plus the lopsided/balanced-evidence scenario
   checks called for in §10.5).
3. **`src/abduction.rs`** — data model, persistence, prompts, and parsing as
   specified in §4, with one deviation: `InvestigateRoles`/role config reuses
   `swarm::Target` directly instead of a duplicate type (simpler, no
   behavior difference). 22 unit tests.
4. **`investigate_agent` + `run_investigation` in `main.rs`** — as specified
   in §5, taking `default_provider: Option<&ActiveProvider>` from the REPL's
   active provider for the zero-setup fallback case (§5's sketch left this
   an implementation detail; passing it explicitly from the call site was
   simpler than re-deriving it inside the function).
5. **`.junebug/investigate.json` role config + `/investigate-setup`** — as
   specified in §6/§7.
6. **Command wiring** — `/investigate`, `/investigate-status`,
   `/investigate-setup`, help text, `SLASH_COMMANDS` (bumped 16 → 19),
   README.
7. **Live pty verification** (§10) — see "What the live test found" below.

### What the live test found

The first live run (`--provider ollama --model qwen3:8b --permission
read-only`, a scratch repo with a deliberately ambiguous CI-slowdown
scenario) aborted on the very first hypothesis-generation call: the model
replied with `"id": 1` (a bare JSON number) instead of a string, despite the
prompt asking for one. `serde_json` correctly refuses to coerce that, and —
notably — the existing one-retry reformat step didn't save it, because the
model made the identical mistake on the retry too. Fixed by making
`RawHypothesis`'s `id` field accept either type via a custom
`deserialize_with` (stringifies a JSON number rather than erroring),
regression-tested with the exact fixture (`"id":1`) that failed live. This
is the kind of gap the plan's §10.4 live-verification step exists to catch —
worth calling out because it was a real, reproducible failure on the first
try, not a hypothetical.

After the fix, a full run completed end-to-end and was inspected directly:
5 categorized hypotheses generated (mundane/systemic/intentional/
measurement-error/contrarian, matching the forced-diversity prompt),
5 blind per-hypothesis evaluations, a skeptic critique pass, and a synthesis
with all five required headings (FACT/INFERENCE/UNCERTAINTY/ALTERNATIVES/
DISCRIMINATOR). The persisted `.junebug/investigations/<id>.json` had
posteriors summing to 1.0 (0.997 before floating-point display rounding),
`/investigate-status` matched the in-run summary exactly, and `git status`
on the scratch repo confirmed zero writes under `--permission read-only` —
confirming the feature's one hard functional requirement that has no
`/swarm` equivalent to copy a test from (§10.4).

One non-bug observation worth recording: the small local model's posteriors
came out nearly uniform (~20% each) rather than clearly favoring the
mundane explanation a stronger model would likely have converged on — the
harness's math was correct given the (weak/hedged) likelihood estimates it
was fed; this is a model-capability ceiling, not a defect in `bayes.rs` or
the orchestration. Worth keeping in mind when picking roles via
`/investigate-setup` for anything that matters: a small local model can run
the harness's mechanics correctly while still producing a mushy result if
its own likelihood judgments are weak.

### A second bug, found live in real use (post-ship)

Field use (a real `/investigate` run with `generator`/`synthesizer` roles
assigned to a `claude-cli`/`codex-cli` delegate via `/investigate-setup`,
on a question whose true answer was "this mechanic doesn't exist in the
codebase") surfaced a bug the local-Ollama live test never could have: the
generator produced no JSON at all, explored the repo on its own, wrote its
own plan file, and replied in plain prose — twice, surviving the one-retry
reformat unchanged. Root cause: `cli_delegate::stream_turn` sends a
delegate provider only the *latest user message*, silently dropping the
system prompt entirely (correct for `/swarm`, whose request builders are
already self-contained; not correct here) — and `generate_request`/
`evaluate_request` originally put the entire JSON-format spec **only** in
`GENERATOR_SYSTEM`/`EVALUATOR_SYSTEM`, so a delegate-routed role received
nothing but the bare question and had no idea a structured reply was
expected. Its confused response ("no schema was ever specified") was
accurate from its own vantage point.

Fixed by moving every format/task instruction into the request text itself
for all four roles (`generate_request`, `evaluate_request`,
`critique_request`, `synthesize_request`), so any provider — including one
that never sees the system message — has everything it needs from the user
turn alone; the `*_SYSTEM` consts are now short role-framing only, with an
explicit note that the request text may be their only source of
instructions. Also fixed a related prompt gap the same incident exposed:
`GENERATOR_SYSTEM` never said what to do with a *negative* finding, so
"this doesn't exist" had no sanctioned way to become valid JSON — the
request text now explicitly says a negative finding is itself a hypothesis
(e.g. `claim: "no such mechanism exists in this codebase"`). Regression
tests assert the format spec and the negative-finding guidance are present
in the request text specifically (not just the system prompt), so this
can't silently regress back to system-prompt-only.

### Deviations from the original spec, and why

- `abduction::InvestigateRoles` reuses `swarm::Target` (already imported in
  `main.rs`) instead of defining a duplicate `Target` struct as §6's sketch
  showed — same shape, no reason to have two.
- `run_investigation`'s zero-setup default-role fallback takes
  `default_provider: Option<&ActiveProvider>` as an explicit parameter from
  the REPL dispatch site (`provider.as_ref()`), rather than re-resolving
  "the active provider" from inside the function — the REPL already has it,
  passing it through is simpler and avoids a second source of truth.
- Evaluator/skeptic/synthesizer JSON-parse failures do **not** get the
  generator's one-retry-reformat treatment in v1; a failed evidence parse
  falls back to leaving that hypothesis's posterior at its prior (logged,
  not fatal) rather than aborting the whole run over one role call. This
  wasn't explicitly specified either way — it follows the spec's general
  principle (§1: "don't let one pass self-certify") toward robustness over
  strictness for the non-generator roles, since a partial result is more
  useful here than an aborted one.

### Not built (still out of scope, per §8 — unchanged)

Parallel evaluators, the evidence-provenance DAG, an active
information-gain planning loop, a conversational spec-authoring front-end,
and the `browser::BrowserMode::Investigations` screen remain unbuilt, as
originally scoped. `bayes::entropy` shipped (per §3) but nothing calls it
yet — it's available for a future info-gain feature, not wired to anything.

---


This document is self-contained: written for an agent with no prior context on this conversation. It specifies a new feature for **Junebug CLI**, a Rust agentic terminal CLI (repo: `weeksdev/junebug-cli`, this checkout at `/Users/andrewweeks/repos/junie_cli`). Read `HANDOFF.md` at the repo root first for the project's overall current state and conventions — this document assumes that context and does not repeat it. `PROJECT_MODE_PLAN.md` specs a *different*, unrelated feature (`/project`, long-running spec-driven execution) — it hasn't been built yet either, but its `run_task_loop` extraction and `src/project.rs` persistence shape are cited below as prior art, not as a dependency. This feature does not require `/project` to exist first.

Build/test loop used throughout this project: `cargo build --release`, `cargo clippy --release --all-targets` (must be clean — the project runs pedantic clippy), `cargo test --release`, then `./install-macos.sh` to install to `~/.local/bin/junebug` for live testing. Every feature in this codebase is live-tested against the actual compiled binary (`expect`-driven pty, not `script(1)` — see `HANDOFF.md`'s REPL testing tips) before being considered done, not just unit-tested.

## 1. Why this exists

Junebug's model calls today are single-pass: one turn, one train of thought, prose straight from evidence to conclusion. For genuinely ambiguous questions ("why does this metric look wrong," "what's the likely cause of this incident," "is this pattern actually suspicious or just noise") a single LLM pass reliably does two things badly: it **anchors on the first plausible explanation** and rationalizes evidence toward it, and it **reports confidence as prose** with no accounting of what was actually weighed.

The fix is not a smarter prompt. It's a harness that forces the model through explicit stages — generate several competing explanations, evaluate each one *blind* to the others, let a separate adversarial pass attack the leading explanation, and compute the actual posterior ranking in deterministic code instead of asking the model to do arithmetic in its head. The model supplies semantic judgment (`P(E|H)` estimates, what a claim means, whether an assumption is hidden); Rust supplies the Bayesian bookkeeping. This mirrors the boss/worker/checker split `/swarm` already uses to prevent a single model from grading its own work — `/investigate` applies the same "don't let one pass self-certify" principle to *reasoning* instead of to *code*.

A key difference from `/swarm`: nothing here needs write access. Generating hypotheses, evaluating evidence, and critiquing a conclusion are all read/reason-only operations, so `/investigate` should work under `read-only` permission and even in plan mode — `/swarm` explicitly cannot (`run_swarm` at `src/main.rs:2549-2558` hard-rejects both). That's a real usability win worth preserving deliberately, not an accident to lose during implementation.

## 2. Reusable building blocks already in this codebase

### `src/swarm.rs` — the pattern to copy, not the code to call

Nothing in `swarm.rs` is reused directly (unlike `/project`'s plan, which reuses `Task`/`SwarmState` verbatim). What's reused is the *shape*:

- Fenced-JSON extraction from a model reply (`parse_tasks`, `src/swarm.rs:119`) — the template for this feature's `parse_hypotheses`/`parse_evidence`.
- Strict last-line parsing for a structured signal for control (`parse_verdict`/`parse_ruling`, `src/swarm.rs:340,373`) — ambiguity always defaults to the stricter/safer outcome. Same discipline applies to `parse_spec_ready`-style detection here if a conversational front-end is ever added (out of scope for v1, see §8).
- One-JSON-file-per-run persistence with a `list`/`load`/`save` API (`state_path`/`load_state`/`save_state`, `src/swarm.rs:166-199`) — `/investigate` gets its own file per investigation, same idea as `/project`'s planned one-file-per-project.
- Deterministic, no-model-call status rendering (`format_status`, `src/swarm.rs:206`) — this feature's `format_summary` equivalent.
- `classify_provider_error`/`retry_delay`/`TRANSIENT_DELAYS`/`RATE_LIMIT_DELAYS` (`src/swarm.rs:237-317`) — reused **directly**, not just as a pattern, by the new turn-runner (§5).

### `src/main.rs::swarm_agent` (`src/main.rs:2157-2416` roughly) — the turn-rendering engine, trimmed down

`swarm_agent` spawns a worker thread running `agent::run_loop` with a `ChannelObserver`, renders `TurnEvent`s live on the main thread (spinner, tool activity, streamed markdown), and handles live single-key controls (`s` status, `p` pause, `Esc`/`Ctrl-C` cancel) via a `SwarmControls` struct that's specific to swarm's long multi-task runs.

`/investigate` doesn't have a multi-task loop to pause mid-way — each role call is one bounded turn. So its runner (§5, `investigate_agent`) is a **smaller copy** of `swarm_agent`'s rendering loop: keep the spinner, keep `Esc`/`Ctrl-C` cancel (`TurnEvent::text/tool/diff` rendering, the `thread::scope` + channel structure), drop the `s`/`p` key handling and the `SwarmControls` dependency entirely. Do not attempt to generalize `swarm_agent` itself to serve both callers in this pass — that's a real refactor (extracting a shared render-loop function parameterized over which keys it handles) worth doing later once both call sites exist and their actual shared surface is obvious, not speculatively now. `swarm.rs`, `run_swarm`, and `swarm_agent` are left **completely untouched** by this feature.

### `agent::run_loop` (`src/agent.rs:109`) — used as-is

Every role call is a `PinnedModel::new(provider, model)` fed into `agent::run_loop` with a fresh `messages: Vec<Value>`, exactly like each `swarm_agent` call today. This is what makes "blind" evaluation free: a fresh message list per call already means an evaluator only sees what you put in its own prompt. No special "hide sibling hypotheses" mechanism is needed beyond *not writing them into that call's request string*.

### `src/cli_delegate.rs` — heterogeneous model families for the skeptic role

The essay's point about decorrelated errors ("assign different epistemic jobs to different model families") maps directly onto `ProviderKind::ClaudeCli`/`CodexCli` (`src/cli_delegate.rs`), which already exist and are wired through the same `ActiveProvider`/`ModelProvider` trait everything else uses. `/investigate-setup` (§6) can assign the skeptic role to a `codex-cli`/`claude-cli` delegate specifically so its adversarial pass runs on a genuinely different model than whatever generated the hypotheses, without this feature needing to know anything about how that provider works internally.

### `src/tool.rs::BUILTIN_TOOLS` / `tool_schemas(plan: bool)` (`src/main.rs:3210`) — reused as-is

Every role in this feature gets the **read-only tool subset** (`tool_schemas(true)`, the same plan-mode-safe set `run_swarm`'s boss-planning phase already uses) so hypothesis generation and evidence evaluation can inspect the workspace/read files/search, but never write or run commands. This is a hard property, not a default: even under `permission = yolo`, `/investigate` should pass `tool_schemas(true)` and the read-only-forced `PolicyEngine`, the same construction plan mode already uses (`policy::PolicyEngine::evaluate` reads a hard plan-mode guard — see `src/policy.rs`). Concretely: build the engine with `plan = true` regardless of the ambient permission mode, for every role, always.

## 3. New module: `src/bayes.rs`

Pure, deterministic, no I/O, no `Value`/serde dependency at all — just math, unit-testable without touching a provider. This is the piece that must never run inside the model.

```rust
//! Deterministic Bayesian bookkeeping. The model estimates likelihoods in
//! natural language / structured JSON; everything in this module is the
//! arithmetic Junebug performs instead of trusting the model to do it.

const EPS: f64 = 1e-6;

/// Natural-log odds of a probability, clamped away from 0/1 so `ln` never
/// sees zero or a negative denominator.
fn logit(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

fn sigmoid(log_odds: f64) -> f64 {
    1.0 / (1.0 + (-log_odds).exp())
}

/// One piece of evidence's effect on one hypothesis: `P(E|H) / P(E|¬H)`.
/// > 1 supports the hypothesis, < 1 contradicts it, ≈ 1 is neutral.
#[must_use]
pub fn likelihood_ratio(given_h: f64, given_not_h: f64) -> f64 {
    given_h.max(EPS) / given_not_h.max(EPS)
}

/// Sequential Bayesian update: `Odds(H|E1..En) = Odds(H) * Π LR_i`,
/// converted back to a probability. This is *per-hypothesis* — it does not
/// know about sibling hypotheses, matching how each evaluator call is blind
/// to them (see §2). Call `normalize` afterward to make a hypothesis set's
/// posteriors comparable/sum-to-one.
#[must_use]
pub fn posterior(prior: f64, likelihood_ratios: &[f64]) -> f64 {
    let log_odds = likelihood_ratios
        .iter()
        .fold(logit(prior), |acc, lr| acc + lr.max(EPS).ln());
    sigmoid(log_odds)
}

/// Rescales a set of independently-computed posteriors so they sum to 1,
/// for display as a ranked belief distribution. A no-op (all zero) input is
/// left untouched rather than divide-by-zero.
pub fn normalize(posteriors: &mut [f64]) {
    let sum: f64 = posteriors.iter().sum();
    if sum > EPS {
        for p in posteriors {
            *p /= sum;
        }
    }
}

/// Shannon entropy in nats, for `H(Hypotheses)` — the input to an
/// expected-information-gain discriminator (§8, deferred to a later pass).
#[must_use]
pub fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > EPS)
        .map(|&p| -p * p.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{entropy, likelihood_ratio, normalize, posterior};

    #[test]
    fn strong_supporting_evidence_raises_posterior_above_prior() {
        let lr = likelihood_ratio(0.8, 0.2); // 4.0
        assert!(posterior(0.3, &[lr]) > 0.3);
    }

    #[test]
    fn contradicting_evidence_lowers_posterior_below_prior() {
        let lr = likelihood_ratio(0.1, 0.6);
        assert!(posterior(0.5, &[lr]) < 0.5);
    }

    #[test]
    fn neutral_evidence_leaves_posterior_unchanged() {
        let lr = likelihood_ratio(0.5, 0.5);
        assert!((posterior(0.4, &[lr]) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn normalize_scales_to_sum_one() {
        let mut values = [0.6, 0.3, 0.1];
        normalize(&mut values);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_is_a_no_op_on_all_zero_input() {
        let mut values = [0.0, 0.0];
        normalize(&mut values);
        assert_eq!(values, [0.0, 0.0]);
    }

    #[test]
    fn entropy_is_zero_for_a_certain_outcome() {
        assert!(entropy(&[1.0, 0.0, 0.0]) < 1e-9);
    }

    #[test]
    fn entropy_is_maximal_for_a_uniform_distribution() {
        let uniform = entropy(&[0.25, 0.25, 0.25, 0.25]);
        let skewed = entropy(&[0.7, 0.1, 0.1, 0.1]);
        assert!(uniform > skewed);
    }
}
```

`src/lib.rs`: add `pub mod bayes;`.

## 4. New module: `src/abduction.rs`

### Data model

```rust
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub observation: String,
    pub given_h: f64,
    pub given_not_h: f64,
    /// Filled in mechanically from the ratio (>1 "supports", <1
    /// "contradicts", ~1 "neutral") — display sugar, not a separate signal.
    pub stance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub claim: String,
    /// One of the Generator's forced diversity categories — "mundane",
    /// "systemic", "intentional", "measurement-error", "contrarian", or a
    /// free-text label if the model adds a sixth. Not enforced by an enum:
    /// the Generator prompt requires the categories, the parser doesn't.
    pub category: String,
    pub prior: f64,
    pub evidence: Vec<Evidence>,
    /// Set by `bayes::posterior` + `bayes::normalize` after the evaluator
    /// call returns. Zero until then.
    pub posterior: f64,
    pub assumptions: Vec<String>,
    pub falsifiers: Vec<String>,
    /// Filled in by the skeptic pass; empty until then.
    pub critique: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Investigation {
    pub id: String,
    pub question: String,
    pub observations: Vec<String>,
    pub hypotheses: Vec<Hypothesis>,
    pub status: Status,
    /// The Synthesizer's final FACT/INFERENCE/UNCERTAINTY/ALTERNATIVES/
    /// DISCRIMINATOR write-up. `None` until the Reviewing phase finishes.
    pub synthesis: Option<String>,
    pub session_log: PathBuf,
    pub created_at: u64,
    pub updated_at: u64,
}
```

### Persistence

One JSON file per investigation at `.junebug/investigations/<id>.json`, directly mirroring `swarm.rs`'s `state_path`/`load_state`/`save_state` (`src/swarm.rs:166-199`) but one-per-run instead of one-per-workspace:

```rust
#[must_use]
pub fn dir(workspace: &Path) -> PathBuf {
    workspace.join(".junebug").join("investigations")
}

/// Kebab-case title + a short FNV-1a suffix over the current-time nanos, so
/// two investigations with the same question don't collide. Mirrors
/// `checkpoint.rs`'s stable-hash approach; collisions only need to be
/// unlikely, not cryptographically impossible.
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
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{trimmed}-{nanos:x}")
}

/// # Errors
/// Returns an error when the investigation cannot be serialized or written.
pub fn save(workspace: &Path, investigation: &mut Investigation) -> Result<(), String> {
    investigation.updated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir(workspace).join(format!("{}.json", investigation.id));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(investigation).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

/// # Errors
/// Returns an error when the file is missing or malformed.
pub fn load(workspace: &Path, id: &str) -> Result<Investigation, String> {
    let path = dir(workspace).join(format!("{id}.json"));
    let text = std::fs::read_to_string(&path)
        .map_err(|_| format!("no investigation named '{id}'"))?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

/// Every investigation in the workspace, newest `updated_at` first.
/// # Errors
/// Returns an error only when the directory exists but cannot be read.
pub fn list(workspace: &Path) -> Result<Vec<Investigation>, String> {
    let path = dir(workspace);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut items = Vec::new();
    for entry in std::fs::read_dir(&path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(entry.path())
            && let Ok(investigation) = serde_json::from_str::<Investigation>(&text)
        {
            items.push(investigation);
        }
    }
    items.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(items)
}
```

### Parsing model replies

Mirrors `swarm::parse_tasks` (`src/swarm.rs:119`) exactly — extract a fenced ` ```json ` block, cap array length, one reformat retry on parse failure at the call site (not inside the parser):

```rust
const MAX_HYPOTHESES: usize = 7;

/// # Errors
/// Returns an error when no valid JSON hypothesis array is found in `text`.
pub fn parse_hypotheses(text: &str) -> Result<Vec<Hypothesis>, String> {
    // identical fenced-block extraction strategy to swarm::parse_tasks;
    // implementer: factor the shared "find ```json ... ``` or bare [...]"
    // scanner out of swarm.rs into a small shared helper (e.g.
    // `src/jsonblock.rs`) rather than copy-pasting it, since this is the
    // second caller.
    todo!()
}

/// # Errors
/// Returns an error when no valid JSON evidence array is found in `text`.
pub fn parse_evidence(text: &str) -> Result<Vec<Evidence>, String> {
    todo!()
}
```

That `jsonblock.rs` extraction is a good first sub-task of Phase 1 (§7): it's a pure refactor of existing, tested code (move, don't rewrite), and having it land *before* `parse_hypotheses` is written means the new parser is built directly on the shared helper instead of a second copy that has to be reconciled later.

### Prompts

Same `pub const &str` + `*_request(...)` builder style as `swarm.rs:387-450`:

```rust
pub const GENERATOR_SYSTEM: &str = "You are the hypothesis-generation agent of an \
abductive reasoning harness. Given observations, generate at least 5 materially \
different causal explanations — do not converge on one early. Include at least: \
one mundane/ordinary explanation, one institutional or systemic explanation, one \
explanation involving intentional action, one explanation based on measurement or \
reporting error, and one explanation that contradicts the apparent narrative. For \
each, give a short claim, a category label, a rough prior in [0,1] (your genuine \
best guess, not a hedge toward 0.5), and any hidden assumptions or falsifiers. \
Reply with a ```json fenced array of objects with fields: id, claim, category, \
prior, assumptions (array of strings), falsifiers (array of strings).";

#[must_use]
pub fn generate_request(question: &str, observations: &[String]) -> String {
    format!(
        "Question: {question}\n\nObservations:\n{}",
        observations
            .iter()
            .map(|o| format!("- {o}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

pub const EVALUATOR_SYSTEM: &str = "You are an evidence-evaluation agent. You are \
shown exactly one hypothesis and a list of observations — you do not know what \
other hypotheses exist, and must not speculate about them. For each observation, \
estimate P(observation | this hypothesis is true) and P(observation | this \
hypothesis is false), both in (0,1]. Be honest: most observations are only weak \
evidence one way or the other — reserve extreme ratios for observations that \
genuinely could not occur, or could only occur, under this hypothesis. Reply with \
a ```json fenced array of objects with fields: observation, given_h, given_not_h.";

#[must_use]
pub fn evaluate_request(hypothesis: &Hypothesis, observations: &[String]) -> String {
    format!(
        "Hypothesis: {}\n\nObservations:\n{}",
        hypothesis.claim,
        observations
            .iter()
            .map(|o| format!("- {o}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

pub const SKEPTIC_SYSTEM: &str = "You are the adversarial red-team agent. You are \
shown the full ranked hypothesis list with its evidence. Your only job is to \
attack the leading hypothesis: name evidence that does not actually support it as \
strongly as claimed, alternative explanations for each supporting observation, \
base-rate neglect, selection bias, confirmation bias, hidden assumptions, omitted \
variables, and causal-direction errors. Do not be diplomatic. If the leading \
hypothesis genuinely holds up, say so plainly and briefly instead of manufacturing \
criticism — false balance is its own failure mode.";

#[must_use]
pub fn critique_request(ranked_summary: &str) -> String {
    format!("Ranked hypotheses with evidence:\n\n{ranked_summary}")
}

pub const SYNTHESIZER_SYSTEM: &str = "You are the final synthesis agent. Given the \
ranked hypotheses, their evidence, and the red-team critique, write the answer for \
a human reader under exactly these headings: FACT (directly established, not \
inferred), INFERENCE (the best-supported explanation and why), UNCERTAINTY (what \
remains genuinely unknown), ALTERNATIVES (other hypotheses that are not ruled out, \
with why they rank lower), DISCRIMINATOR (what evidence would most change this \
conclusion if found). Do not present the numeric posteriors as more precise than \
they are — describe confidence qualitatively (e.g. \"clearly favored\", \"a close \
call between H1 and H3\"), not as a bare percentage.";

#[must_use]
pub fn synthesize_request(ranked_summary: &str, critique: &str) -> String {
    format!("Ranked hypotheses:\n\n{ranked_summary}\n\nRed-team critique:\n\n{critique}")
}

/// `swarm::format_status`-equivalent: deterministic, no model call. Used by
/// both `/investigate-status` (plain text) and the browser's detail pane
/// (§8, deferred).
#[must_use]
pub fn format_summary(investigation: &Investigation) -> String {
    let mut ranked = investigation.hypotheses.clone();
    ranked.sort_by(|a, b| b.posterior.partial_cmp(&a.posterior).unwrap());
    let mut out = format!("{}\n", investigation.question);
    for h in &ranked {
        out += &format!("  {:>5.1}%  [{}] {}\n", h.posterior * 100.0, h.category, h.claim);
    }
    out
}
```

`src/lib.rs`: add `pub mod abduction;`.

## 5. Orchestration in `src/main.rs`

New function `run_investigation`, structurally parallel to `run_swarm` (`src/main.rs:2537`) but simpler — no per-task loop, no resume, no pause. Sequential phases, saved to disk after each one (crash safety, same reasoning as swarm saving `SwarmState` after planning):

```rust
#[allow(clippy::too_many_arguments)]
fn run_investigation(
    question: &str,
    root: &Path,
    workspace: &Workspace,
    permission: PermissionMode,
    max_context_chars: usize,
    main_session: &SessionWriter,
) {
    // Every role is forced read-only regardless of ambient permission — see
    // §2's note on tool_schemas(true) + a hard plan=true PolicyEngine.
    let roles = investigate::load_roles(root).unwrap_or_default(); // §6
    let build = |target: &investigate::Target| -> Result<ActiveProvider, String> {
        let kind = ProviderKind::parse(&target.provider)?;
        ActiveProvider::from_environment(kind, Some(target.model.clone()), root, permission, true)
    };
    let (Ok(generator), Ok(evaluator), Ok(skeptic), Ok(synthesizer)) =
        (build(&roles.generator), build(&roles.evaluator), build(&roles.skeptic), build(&roles.synthesizer))
    else {
        eprintln!("{RED}error:{RESET} could not start one or more investigation roles");
        return;
    };

    let session = match SessionWriter::create(root) {
        Ok(s) => s,
        Err(error) => { eprintln!("{RED}error:{RESET} {error}"); return; }
    };
    let _ = session.record("investigate_question", question);
    let _ = main_session.record("investigate_session", &session.path().display().to_string());

    let tools = tool_schemas(true); // read-only subset, always
    let policy = PolicyEngine::new(permission, true); // plan=true forces read-only

    // Phase 1: generate.
    let mut investigation = abduction::Investigation {
        id: abduction::slug_for(question),
        question: question.to_owned(),
        observations: Vec::new(), // populated from the request text / read-tool results the Generator surfaces — see note below
        hypotheses: Vec::new(),
        status: abduction::Status::Generating,
        synthesis: None,
        session_log: session.path().to_owned(),
        created_at: now_secs(),
        updated_at: now_secs(),
    };
    let generated = investigate_agent(&generator, roles.generator.model.as_str(),
        abduction::GENERATOR_SYSTEM, &abduction::generate_request(question, &investigation.observations),
        &tools, &policy, workspace, &session, max_context_chars, true)?;
    investigation.hypotheses = abduction::parse_hypotheses(&generated)?;
    let _ = abduction::save(&workspace.root(), &mut investigation);

    // Phase 2: blind per-hypothesis evaluation, sequential (see §2 on why
    // not parallel), fresh message list per call.
    investigation.status = abduction::Status::Evaluating;
    for hypothesis in &mut investigation.hypotheses {
        let reply = investigate_agent(&evaluator, roles.evaluator.model.as_str(),
            abduction::EVALUATOR_SYSTEM, &abduction::evaluate_request(hypothesis, &investigation.observations),
            &tools, &policy, workspace, &session, max_context_chars, false)?;
        hypothesis.evidence = abduction::parse_evidence(&reply)?;
        let ratios: Vec<f64> = hypothesis.evidence.iter()
            .map(|e| bayes::likelihood_ratio(e.given_h, e.given_not_h)).collect();
        hypothesis.posterior = bayes::posterior(hypothesis.prior, &ratios);
    }
    let mut posteriors: Vec<f64> = investigation.hypotheses.iter().map(|h| h.posterior).collect();
    bayes::normalize(&mut posteriors);
    for (h, p) in investigation.hypotheses.iter_mut().zip(posteriors) { h.posterior = p; }
    let _ = abduction::save(&workspace.root(), &mut investigation);

    // Phase 3: adversarial pass over the ranked summary, then synthesis.
    investigation.status = abduction::Status::Reviewing;
    let summary = abduction::format_summary(&investigation);
    let critique = investigate_agent(&skeptic, roles.skeptic.model.as_str(),
        abduction::SKEPTIC_SYSTEM, &abduction::critique_request(&summary),
        &tools, &policy, workspace, &session, max_context_chars, true)?;
    let synthesis = investigate_agent(&synthesizer, roles.synthesizer.model.as_str(),
        abduction::SYNTHESIZER_SYSTEM, &abduction::synthesize_request(&summary, &critique),
        &tools, &policy, workspace, &session, max_context_chars, true)?;
    investigation.synthesis = Some(synthesis);
    investigation.status = abduction::Status::Done;
    let _ = abduction::save(&workspace.root(), &mut investigation);
}
```

(Treat this as a build-order sketch, not final code — error propagation, the exact `investigate_agent` signature, and where `observations` actually comes from — a first Observer/fact-extraction pass over pasted text or read-tool output — are implementation details for Phase 2, §7. The one hard constraint: **the Bayesian math above happens in `bayes::`, never in a prompt asking the model to compute a posterior.**)

`investigate_agent` is the trimmed-down `swarm_agent` copy described in §2 — same `thread::scope` + `ChannelObserver` + spinner structure, no `SwarmControls`, `Esc`/`Ctrl-C` cancel only, reusing `swarm::classify_provider_error`/`retry_delay` for the same transient/rate-limit retry budgets `swarm_agent` already uses.

## 6. `.junebug/investigate.json` — optional role assignment

Mirrors `.junebug/swarm.json` (`swarm::load`/`save`, `src/swarm.rs:53-96`), but **optional** — this is the deliberate UX difference from `/swarm`, which hard-requires `/swarm-setup` first (`src/main.rs:2561-2564`). `/investigate` should work with zero setup by defaulting every role to the REPL's currently active provider/model:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target { pub provider: String, pub model: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigateRoles {
    pub generator: Target,
    pub evaluator: Target,
    pub skeptic: Target,
    pub synthesizer: Target,
}

/// Loads `.junebug/investigate.json` if present. Returns `None` (not an
/// error) when absent — the caller falls back to the active provider/model
/// for every role, which is what makes `/investigate` work with no setup.
pub fn load_roles(workspace: &Path) -> Result<Option<InvestigateRoles>, String> { /* mirrors swarm::load */ }
pub fn save_roles(roles: &InvestigateRoles) -> Result<PathBuf, String> { /* mirrors swarm::save */ }
```

`/investigate-setup` (optional command, same interactive picker `/swarm-setup` uses) lets a user assign, e.g., the skeptic role to `codex-cli` while everything else stays on the active cloud provider — the heterogeneous-model-family case from §2.

## 7. Command wiring (mechanical, same pattern every prior slash command followed)

- `src/main.rs` REPL dispatch: add an `"investigate" | "investigation" =>` arm next to `"swarm" =>` (`src/main.rs:944`), parsing the rest of the line as the question (usage: `/investigate <question>`, error text mirrors `/swarm`'s `argument.is_empty()` branch at `src/main.rs:945-946`).
- `/investigate-status [id]` mirrors `handle_swarm_status` (`src/main.rs:2416`) — bare form shows the most recently updated investigation; add `id` to show a specific one. Plain text via `abduction::format_summary`, no TUI.
- `/investigate-setup` mirrors the `/swarm-setup` dispatch arm — optional, see §6.
- `/help` text block (search `"help" => eprintln!` in `main.rs`) — add `/investigate` lines.
- `src/editor.rs::SLASH_COMMANDS` (`src/editor.rs:12`) — add `("/investigate", "...")`, `("/investigate-status", "...")`; bump the declared array length from 16 (the compiler will tell you the new N).
- `README.md` — add to the slash-command list, same style as the existing `/swarm` entry.
- `src/lib.rs` — `pub mod bayes;`, `pub mod abduction;`.

## 8. Explicitly out of scope for v1 (named so they aren't silently forgotten)

1. **True parallel evaluator execution.** Same reasoning `/project`'s spec gives for staying sequential: one live terminal render stream at a time is simpler and avoids a second, unrelated concurrency problem. Sequential per-hypothesis evaluation is fine — these are short calls, not long task loops.
2. **Full evidence-provenance DAG** (source → observation → interpretation → evidence → hypothesis → conclusion, with individually-challengeable edges). v1 stores evidence as a flat list per hypothesis. A real graph structure is separate follow-up work, genuinely valuable but a materially bigger data model and UI.
3. **Expected-information-gain planner as an active tool.** `bayes::entropy` ships in v1 (cheap, useful on its own), but a loop that computes `argmax IG(E)` over *candidate* future observations and drives new tool calls to go find them does not. v1's Synthesizer produces a DISCRIMINATOR section as prose, not a computed ranking.
4. **A conversational spec-authoring front-end** (something like `/project new`'s back-and-forth before committing to hypotheses). v1's `/investigate <question>` takes the question and whatever's in context/pasted directly; a dedicated clarifying-questions pass is a natural v2, not required to ship something useful.
5. **A browser screen** (`browser::BrowserMode::Investigations`, modeled on `run_commits`/the planned `run_projects`). Useful once there's history worth browsing; `/investigate-status` covers the v1 need.
6. **Showing bare numeric posteriors to the user by default.** The persisted JSON always has them (for `/investigate-status`, debugging, tests); the Synthesizer's prose is deliberately qualitative — see the essay's own caution about displaying LLM-adjacent numbers as falsely precise, and `SYNTHESIZER_SYSTEM`'s explicit instruction above.

## 9. Suggested build order

1. Extract the shared fenced-JSON-array scanner out of `swarm::parse_tasks` into `src/jsonblock.rs` (or similar) as a pure refactor — no behavior change, regression-test `/swarm <goal>` still parses tasks identically. Small, safe, unblocks step 3 cleanly.
2. `src/bayes.rs` + unit tests (§3). No dependents, safe to land alone, most of the "did I get the epistemology right" risk is concentrated and testable here in isolation.
3. `src/abduction.rs`: data model, persistence (`slug_for`/`save`/`load`/`list`), `parse_hypotheses`/`parse_evidence` (built on step 1's shared scanner), prompts, `format_summary`. Unit tests for JSON round-trip, slug generation, and parsing against hand-written fixture replies (mirror `swarm.rs`'s test style).
4. `investigate_agent` (trimmed `swarm_agent` copy, §2/§5) + `run_investigation` (§5). This is the step that needs a live provider to really exercise.
5. `.junebug/investigate.json` role config + `/investigate-setup` (§6) — can ship after step 4 works with the active-provider default; not a blocker for a useful v1.
6. Command wiring (§7): `/investigate`, `/investigate-status`, help text, `SLASH_COMMANDS`, README.
7. Live pty verification (§10).

## 10. Verification

1. `cargo build --release && cargo clippy --release --all-targets` clean at every step above, not just at the end.
2. `cargo test --release` — new unit tests for `bayes.rs` and `abduction.rs` pass alongside everything existing (`swarm.rs`'s tests must be unaffected — nothing in `swarm.rs` is modified by this feature).
3. Confirm `/swarm <goal>` behaves identically after step 1's refactor (same task-parsing behavior, same `/swarm-status` output) — this is the one step that touches code `/swarm` depends on.
4. Live end-to-end pty test (`expect`, not `script` — see `HANDOFF.md`) once step 6 lands: start a scratch repo with an obviously ambiguous scenario seeded into a file or pasted as context, run `/investigate <question>`, confirm it produces ≥5 categorized hypotheses, a populated `.junebug/investigations/<id>.json` with non-zero posteriors that sum to ~1, a critique, and a synthesis with all five required headings. Confirm `/investigate-status` matches. Confirm it runs successfully under `--permission read-only` and under `--plan` — this is the feature's one hard functional requirement that has no equivalent in `/swarm` to copy a test from, so it needs its own explicit check.
5. Sanity-check the Bayesian math isn't just plausible-looking: construct one fixture where the evidence is overwhelmingly one-sided and confirm the winning hypothesis's posterior actually lands high (>0.8, say), and one balanced-evidence fixture where no hypothesis dominates — both as `bayes.rs`/`abduction.rs` unit tests, not just eyeballed from a live run.
