# `/project` mode — implementation spec for handoff

This document is self-contained: written for an agent with no prior context on this conversation. It specifies a new feature for **Junebug CLI**, a Rust agentic terminal CLI (repo: `weeksdev/junebug-cli`, this checkout at `/Users/andrewweeks/repos/junie_cli`). Read `HANDOFF.md` at the repo root first for the project's overall current state and conventions — this document assumes that context and does not repeat it. `README.md` documents user-facing behavior; `PLAN.md` is the original product roadmap (unrelated to this feature, do not confuse the two).

Build/test loop used throughout this project: `cargo build --release`, `cargo clippy --release --all-targets` (must be clean — the project runs pedantic clippy), `cargo test --release`, then `./install-macos.sh` to install the built binary to `~/.local/bin/junebug` for live testing. Every feature in this codebase has been live-tested against the actual compiled binary (including via a pty-driven Python script for full-screen TUI screens — see "Verification" at the end) before being considered done, not just unit-tested.

## 1. Why this exists

Junebug already has `/swarm`: a boss/worker/checker model orchestration loop (one expensive "boss" model plans and reviews, cheap "worker" models do the work, a "checker" model independently verifies each task before it counts as done, disputes escalate to the boss). This is the right foundation for giving the CLI a real "give it a project idea and it does the work" mode — but today it's limited:

- One active run per workspace (`.junebug/swarm_state.json`, singular).
- No spec-authoring step — the boss free-plans from a single goal string typed on the command line, no back-and-forth to nail down what's actually wanted.
- No cost visibility at all (see section 3).
- No way to browse past/current runs — `/swarm-status` reads the one state file that exists right now; there is no history.

The product goal (verbatim from the design discussion that produced this spec): a `/project` concept that is **long-running but explicitly not open-ended** — contrast with "always-on" autonomous agents like OpenClaw, which run indefinitely against a vague heartbeat checklist with broad system access and no defined completion state (and have documented security/privacy concerns for exactly that reason: unbounded runtime + broad access + no verification gate). `/project` instead: you have a conversation to agree a concrete spec with defined tasks and a checker-verified definition of done, agents then execute that spec for as long as it takes (which may be a long time — that's fine, it's bounded by the spec, not by a session), and you can check in on cost and progress at any point via a dedicated browser screen — never a black box, never running forever.

**Read this whole document before writing code.** Section 8 lists what is deliberately *not* in scope for this pass, and why — those aren't oversights.

## 2. Reusable building blocks already in this codebase

### `src/swarm.rs` — reuse directly, unchanged

- `Task { id: usize, title: String, instructions: String, check: String }` (Serialize/Deserialize) — one unit of work.
- `SwarmRoles { boss: Target, worker: Target, checker: Target }` where `Target { provider: String, model: String }`.
- `parse_tasks(text: &str) -> Result<Vec<Task>, String>` — extracts a fenced ` ```json ` task array (or a bare `[...]`) from a model reply, caps at `MAX_TASKS = 12`, renumbers ids.
- `constitution_of(plan: &str) -> String` — everything before the JSON block; the "definition of done-right" prose.
- `format_status(state: &SwarmState, phase: Option<&str>) -> String` — deterministic, no-model-call progress readout (✓/✗/· per task). Used today by `/swarm-status`; the new project browser needs an equivalent for `Project` (see section 4).
- Prompts: `BOSS_PLAN_SYSTEM`, `WORKER_SYSTEM`, `CHECKER_SYSTEM`, `BOSS_RULING_SYSTEM`, `BOSS_REVIEW_SYSTEM`, plus their `*_request(...)` builders. Reused as-is by project execution.
- `parse_verdict`/`Verdict`, `parse_ruling`/`Ruling` — strict parsing of the checker's `VERDICT: PASS|FAIL: <reason>` and the boss's `RULING: WORKER|CHECKER: <guidance>` lines. Silence/ambiguity always defaults to the stricter outcome (fail / uphold checker) — preserve this property in anything new.
- `classify_provider_error`/`retry_delay`/`TRANSIENT_DELAYS`/`RATE_LIMIT_DELAYS` — shared retry policy for provider turns during a swarm run.
- `SwarmState`, `state_path`, `load_state`, `save_state`, `clear_state` — the existing single-run-per-workspace persistence. **Leave these completely untouched.** `/project` is additive; plain `/swarm <goal>` must keep working exactly as it does today, unmodified in behavior.

### `src/main.rs::run_swarm` (~line 2382 to ~2828) — the actual orchestration engine

Structure, precisely:

1. Guards: refuses in plan mode or `ReadOnly` permission (workers need write access).
2. Loads `SwarmRoles` via `swarm::load(root)`; builds three `ActiveProvider`s via `ActiveProvider::from_environment(kind, Some(model), root, permission, plan_mode)`.
3. Creates a **fresh `SessionWriter`** for the run (`SessionWriter::create(root)`) — the swarm's own conversation history is isolated from the main REPL session; only a pointer (`swarm_session` event) and the final outcome get written back to the main session. **This is the pattern to copy for per-project session logs.**
4. Phase 1 — boss plans (or, on `/swarm resume`, loads `SwarmState` from disk instead of re-planning): `swarm_agent(&boss, model, BOSS_PLAN_SYSTEM, plan_request(goal), plan_tools, boss_policy, ...)` → `swarm::parse_tasks`. One retry if the first reply doesn't parse. Saves `SwarmState` to disk immediately after planning (so even planning survives a crash).
5. Phase 2 — **the actual per-task loop**, `for task in &tasks { ... }` (~line 2615–2792): skip already-finished tasks (resume support), check `controls.pause` before starting each task, run a local `verify` closure (worker attempt → checker verdict) up to `swarm::MAX_ATTEMPTS` times, and on repeated failure escalate to the boss for a `Ruling` (which can overrule the checker or grant one final guided rework). Saves `SwarmState` to disk after every single task (`swarm::save_state`), which is what makes `/swarm resume` safe.
6. Phase 3 — boss final review (`BOSS_REVIEW_SYSTEM`) against the full workspace diff, then `swarm::clear_state(root)` (the run is done, no more resume needed), and the outcome gets pushed into the **main REPL's** `messages` as a synthetic user/assistant pair so follow-up chat is informed.

`swarm_agent` (a separate function in main.rs, not shown above — search for `fn swarm_agent`) is the thing that actually runs one agent turn: spawns a worker thread running `agent::run_loop` with a `ChannelObserver`, renders its `TurnEvent`s live on the main thread (spinner, tool activity, text), and returns **`Result<String, String>`** — just the final assistant text. It discards the `agent::LoopOutcome` it gets back from `run_loop`, which is the actual problem for cost tracking (see section 3).

There are exactly **three textual call sites** of `swarm_agent` in `run_swarm`: once inside the `boss_run` closure (itself called 4 times at runtime: initial plan, retry-reformat plan, dispute ruling, final review), and twice inside the `verify` closure (worker, then checker; `verify` itself is called up to `MAX_ATTEMPTS` times per task, plus once more for the post-ruling guided rework).

### `src/browser.rs` — the UI template (`/commits`, shipped just before this feature)

This is the concrete pattern to copy for the project browser screen:

- `BrowserMode` enum (`Explorer`, `Changes`, `Commits`) drives `draw_screen`'s header/footer text and whether the detail pane is diff-colored (`color_diff`) or plain (`fit_ansi`). **Add a `Projects` variant here.**
- `draw_screen(title, root, labels, selected, detail_title, detail, detail_scroll, query, searching, mode, detail_focused, notice)` — the shared two-pane (list + detail) renderer used by all three existing screens. List labels are **plain text only** (they go through `fit()`, which treats ANSI escapes as control characters and mangles them — never put color codes in a `labels` entry).
- Each screen's `run_*` function is a self-contained raw-mode event loop: `terminal::enable_raw_mode()` + `enter_screen()` at the top, `leave_screen()` + `disable_raw_mode()` at the bottom, `event::read()` per iteration, `q`/`Esc` to quit. `run_commits` is the closest template: a list with a cached, lazily-fetched preview pane (`cached_hash`/`cached_lines`, refetched only when the selection changes), `/` to search-filter, and Enter drilling into **a different, fully self-contained nested screen** (`run_changes`) by explicitly unwinding first — `leave_screen(); disable_raw_mode();` **before** calling the nested screen, then `enable_raw_mode(); enter_screen();` **after** it returns. This matters: most terminals implement the alt-screen toggle (`\x1b[?1049h/l`) as a single-slot toggle, not a real stack, so entering a second alt-screen without unwinding the first one first strands the outer screen when the inner one exits. `edit_file` (opening `$EDITOR` from `/explorer`) uses the identical unwind/rewind pattern. **The project browser's "resume this project" action should use exactly this same unwind pattern** when it drops into the (headless, non-full-screen) execution output.
- `enter_screen()`/`leave_screen()`/`view_height()`/`fit`/`fit_ansi`/`sanitize`/`char_columns` are shared low-level helpers, reuse them.

### `src/checkpoint.rs` — the concurrency constraint that shapes section 8

One shadow Git repository per **workspace** (not per project, not per task) at `~/.junebug/checkpoints/<workspace-id>`, used as the mutating-tool-call safety net for `/rewind`. It is a single shared repo — there is no per-task or per-worker isolation. This is why concurrent/parallel task execution is out of scope for this pass (section 8): two workers checkpointing at once would race on that one shadow repo's index/HEAD, on top of both potentially touching overlapping files in the one real working tree.

### `src/provider.rs` — token accounting exists; cost does not

`ModelTurn { text_deltas, tool_calls, assistant_message, input_tokens: u32, output_tokens: u32 }` — every provider turn already reports token counts. `agent::LoopOutcome` aggregates these across a whole `run_loop` call. **Nothing in this codebase computes a dollar cost from these today** — confirmed via `grep -ri "cost\|pricing"` across `src/*.rs`: zero hits. This is new ground, specified in section 3.

## 3. New module: `src/cost.rs`

Best-effort USD estimation from token counts — explicitly an *estimate*, never presented as the only source of truth (always show raw token counts alongside it in any UI).

```rust
/// $ per million tokens. Looked up by (provider, model) with prefix
/// matching on the model name, so e.g. "claude-sonnet-5-20260115" matches
/// a "claude-sonnet-5" table entry without hardcoding every dated snapshot.
pub struct ModelPrice {
    pub input_per_million: f64,
    pub output_per_million: f64,
}

/// One accounted turn: which role/provider/model, how many tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub role: String,     // "planner" | "boss" | "worker" | "checker"
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Sum of known-priced entries in USD. Entries for providers/models with no
/// table entry (local Ollama, claude-cli, codex-cli — these have no
/// meaningful per-token price to Junebug) are excluded from the dollar sum
/// but their tokens still count in any token total the caller also shows.
pub fn estimate_usd(entries: &[CostEntry]) -> f64 { /* ... */ }

/// Whether every entry had a known price (for UI: show "~$x.xx" vs
/// "~$x.xx + N tokens on models with no price data").
pub fn all_priced(entries: &[CostEntry]) -> bool { /* ... */ }
```

Seed the pricing table with whatever a quick check of current published Anthropic/OpenAI/DeepSeek/OpenRouter pricing shows at implementation time (prices change; don't trust any number already in this document). Unpriced entries (`ollama`, `local-openai`, `claude-cli`, `codex-cli`) must resolve to "no price data," not `$0` silently indistinguishable from "actually free" — surface that distinction in the UI text (e.g. `~$1.20 + 40K tokens on local models`).

Unit tests: known-model lookup, prefix matching, unknown-model exclusion, `all_priced` correctness. Follow the existing table-driven test style in `swarm.rs`'s `#[cfg(test)] mod tests`.

## 4. New module: `src/project.rs`

### Data model

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectStatus { Draft, Running, Paused, Done, Failed }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,                     // slug: kebab-case title + short suffix
    pub title: String,                  // short human title (planner generates it)
    pub status: ProjectStatus,
    pub roles: SwarmRoles,              // reuses swarm::SwarmRoles
    pub goal: String,                   // original free-text goal
    pub constitution: String,           // from swarm::constitution_of
    pub tasks: Vec<Task>,               // reuses swarm::Task
    pub outcomes: Vec<(usize, String)>, // same shape SwarmState uses: (task id, "done"|"FAILED")
    pub reworks: usize,
    pub failures: usize,
    pub cost: Vec<CostEntry>,           // from cost.rs
    pub budget_usd: Option<f64>,        // optional cap agreed during spec authoring
    pub session_log: PathBuf,           // this project's own session .jsonl
    pub created_at: u64,                // unix seconds
    pub updated_at: u64,
}
```

### Persistence

One JSON file per project at `.junebug/projects/<id>.json` (parallel to the existing single `.junebug/swarm_state.json`, just one-per-project instead of one-per-workspace). Functions to add:

- `list(workspace: &Path) -> Result<Vec<Project>, String>` — read every `.junebug/projects/*.json`, newest `updated_at` first.
- `load(workspace: &Path, id: &str) -> Result<Project, String>`
- `save(workspace: &Path, project: &Project) -> Result<(), String>` — also bumps `updated_at`.
- `slug_for(title: &str) -> String` — kebab-case + short random/hash suffix so two projects with the same title don't collide (mirror `checkpoint.rs`'s `stable_workspace_id` FNV-1a approach for the suffix, or just a short timestamp component — keep it simple and deterministic-enough, collisions just need to be *unlikely*, not cryptographically impossible).
- `format_summary(project: &Project) -> String` — a `swarm::format_status`-equivalent for `Project`: task checklist with ✓/✗/· markers plus a cost line. Used by both `/project status` (plain text) and the browser's detail pane.

### Planner prompt and spec-ready detection

```rust
pub const PLANNER_SYSTEM: &str = "You are the planning agent for a new Junebug project. \
Have a real back-and-forth with the user to understand exactly what they want built, asking \
clarifying questions about scope, constraints, and what \"done\" looks like — do not rush to a \
spec. Only once you are confident you understand the goal, and the user seems satisfied, reply \
with: a short title, a CONSTITUTION (numbered list of standards defining done-right), a \
```json fenced task array (objects with \"title\", \"instructions\", \"check\", same shape \
worker/checker tasks always use), and optionally a suggested USD budget if the scope implies \
one. End that final message with exactly one line: SPEC: READY";
```

`parse_spec_ready(text: &str) -> bool` — checks for a trailing `SPEC: READY` line (mirror the strict last-line parsing already used by `parse_verdict`/`parse_ruling` in `swarm.rs`: only trust the line, don't infer readiness from prose). When true, the caller runs `swarm::parse_tasks` + `swarm::constitution_of` on the same text (already proven parsers, reused as-is) to build the `Project`.

## 5. Wiring the cost ledger through `run_swarm` / `swarm_agent`

Change `swarm_agent`'s return type from `Result<String, String>` to `Result<(String, u32, u32), String>` (text, input_tokens, output_tokens — pull these straight from the `LoopOutcome` `run_loop` already returns inside `swarm_agent`, currently discarded). Update all three call sites in `run_swarm` (the `boss_run` closure body, and the two calls inside `verify`) to destructure the tuple. For plain `/swarm`, the extra token counts can simply be ignored (`let (text, _, _) = ...`) if you don't want to add a cost ledger to `SwarmState` in this pass — that's fine, `/swarm` doesn't need cost tracking, only `/project` does. Do not change `SwarmState`'s on-disk shape.

**Do this change first, in isolation, and confirm `/swarm <goal>` still behaves identically before building anything else on top of it** — see the regression check in section 9.

## 6. Extracting the shared execution loop

`run_swarm`'s Phase 2 (`for task in &tasks { ... }`, ~line 2615–2792 as described in section 2) is the actual engine a project needs. Extract it into a standalone function both `run_swarm` and the new project runner call, rather than duplicating ~150 lines of tuned retry/pause/escalation logic:

```rust
fn run_task_loop(
    tasks: &[Task],
    constitution: &str,
    roles: &SwarmRoles,
    boss: &ActiveProvider,
    worker: &ActiveProvider,
    checker: &ActiveProvider,
    worker_tools: &[Value],
    checker_tools: &[Value],
    workspace: &Workspace,
    session: &SessionWriter,
    checkpointer: Option<&Checkpointer>,
    max_context_chars: usize,
    controls: &mut SwarmControls,
    already_finished: &[usize],           // task ids already done (resume support)
    mut reworks: usize,
    mut failures: usize,
    on_task_done: &mut dyn FnMut(usize, &str, u32, u32, u32, u32), // task id, "done"|"FAILED", worker/checker/boss token deltas — shape this to whatever the two callers actually need to persist
) -> Result<TaskLoopSummary, String>   // { outcomes: String (rendered), reworks: usize, failures: usize }
```

(Treat the exact callback signature above as a starting sketch, not gospel — the real constraint is: **`run_swarm` refactored to call this must produce byte-for-byte identical behavior to today**, including the exact `eprintln!`/session-record/`save_state` calls that currently happen inline in the loop. The cleanest way to guarantee that is to move the loop body verbatim and thread `SwarmState`-saving through the callback, i.e. `run_swarm`'s callback calls `swarm::save_state` exactly where the loop does today; the project runner's callback calls `project::save` instead.)

`run_swarm` keeps its own copy of "Phase 1: boss plans" and "Phase 3: boss final review" unchanged (both are short and not shared with project execution the same way — see section 7's note on whether project review should also push into the main conversation).

## 7. `/project new` — spec authoring conversation

A small, dedicated conversational loop in `main.rs`, structurally modeled on `run_interactive_turn`/`repl` (not a new UI paradigm — same turn-by-turn model call, same `ChannelObserver` live rendering) but scoped:

- Its own **fresh `messages: Vec<Value>`** and its own fresh `SessionWriter` — spec drafting must not clutter the main chat history, same reasoning as why `/swarm` gets its own session file today (section 2).
- System prompt: `project::PLANNER_SYSTEM`.
- Loop: read user input, run one turn (reuse `agent::run_loop` or the same turn machinery `run_interactive_turn` uses), print the assistant's reply. After each assistant reply, check `project::parse_spec_ready`. If true: parse the spec (`swarm::parse_tasks` + `swarm::constitution_of` + whatever title/budget extraction you add), show it to the user, and ask for an explicit y/N confirmation (reuse the existing approval-prompt pattern already used elsewhere in `main.rs`, e.g. `parse_approval_answer`) before saving. If the user says no, tell the planner agent to keep refining (push a synthetic message like "the user wants changes" and continue the loop) rather than discarding the conversation.
- On accept: build a `Project` in `Draft` status via `project::save`, print its `id`, and tell the user how to run it (`/project run <id>`).
- Should read-only tools be available to the planner (to inspect the workspace while asking questions)? Yes — reuse `tool_schemas(true)` (the same plan-mode-safe read-only subset `run_swarm`'s boss planning phase already uses) so the planner can look around before proposing a spec, without being able to write anything during the conversation.

## 8. `/project run [id]` — execution

- Loads the named `Project`, or (if `id` omitted) the most recently updated one with status `Draft` or `Paused`.
- Builds the three role providers exactly like `run_swarm` does (`ActiveProvider::from_environment`).
- Sets status to `Running`, saves.
- Calls the shared `run_task_loop` (section 6) with a callback that updates `project.outcomes`/`reworks`/`failures`/`cost` and calls `project::save` after every task — same on-disk-survives-a-crash guarantee `/swarm resume` already relies on, now scoped per project.
- **Budget enforcement**: after each task's callback runs, compare `cost::estimate_usd(&project.cost)` against `project.budget_usd` (if set). If exceeded, set `controls.pause = true` (the exact same mechanism `/swarm`'s live `p`/Esc controls already use) so the loop stops cleanly after the current task instead of a new stop mechanism.
- On completion: status → `Done` (or `Failed` if you decide the boss's final review should be able to flag that — simplest v1: always `Done` once the review runs, `Failed` is reserved for a hard abort/error case, not a quality judgment call).
- On interruption/pause/error: status → `Paused`, resumable later by running `/project run <id>` again. **No background process, no daemon** — this is what keeps "long-running" bounded and safe rather than an always-on service; see section "why this exists."
- Open design question left to the implementer: should a finished project's boss review get pushed into the *main* REPL conversation the way `/swarm`'s does today (section 2, Phase 3)? Recommendation: **no, not by default** — a project may finish while the user is doing something else entirely; injecting its result into whatever the main conversation happens to be about is surprising. Instead, print a completion notice to the terminal (`eprintln!`) and rely on the project browser (section 9) as where the user actually goes to read the outcome. If the user is actively watching the terminal when it finishes, that's what the live `ChannelObserver` rendering (already reused from `swarm_agent`) is for.

## 9. `/project` / `/projects` — the browser screen

- New `browser::BrowserMode::Projects` variant (section 2).
- `run_projects(root: &Path)`, modeled directly on `run_commits`: left pane lists projects (`{status icon} {title}  {n}/{m} tasks  ~${cost}`), right pane previews the selected project via `project::format_summary` (task checklist + cost breakdown by role, in the same plain-text style `swarm::format_status` already renders — no ANSI in the label/preview text, same constraint as section 2's `draw_screen` note).
- `/` search-filters by title/goal, same as `/commits`.
- Enter on a `Draft`/`Paused` project: offer to run it — unwind the alt-screen (`leave_screen(); disable_raw_mode();`), call the `/project run` execution path (section 8) so its live activity renders normally on the plain terminal (not inside the alt-screen browser — the existing `swarm_agent` rendering is not itself an alt-screen UI, it's normal scrolling terminal output, so this is a clean fit), then re-enter the browser (`enable_raw_mode(); enter_screen();`) when it returns. This is the same unwind pattern `/commits` uses to drop into `run_changes`, and `edit_file` uses to drop into `$EDITOR` (section 2) — follow it exactly, do not invent a new nesting approach.
- Enter on a `Done`/`Failed` project: just show the detail (already visible in the preview pane; a dedicated full task-by-task drill-in screen is optional polish, not required for v1 — `format_summary`'s checklist rendering in the existing preview pane is enough to be useful).
- `/project status [id]` — plain-text variant mirroring `/swarm-status` for scripting/quick checks without opening the full-screen browser (calls `project::format_summary` directly, no TUI).

### Command wiring (mechanical, same pattern every prior slash command followed)

- `src/main.rs`: REPL command dispatch `match name { ... }` block — add `"project" | "projects" => { ... }` alongside the existing `"changes"`/`"commits"` arms (search for `"commits" | "log" =>` in `main.rs` to find the exact spot). Needs argument parsing for `new`/`run [id]`/`status [id]`/bare (browser).
- `/help` text block (search `"help" => eprintln!` in `main.rs`) — add a `/project` line.
- `--help` REPL summary line (search `REPL: /help /model ...` in `main.rs`) — add `/project`.
- `src/editor.rs`'s `SLASH_COMMANDS` const array — add `("/project", "...")`, bump the array's declared length by one (it's a fixed-size array, e.g. `[(&str, &str); N]` — the compiler will tell you the new N).
- `README.md` — add to the slash-command list paragraph, same style as the existing `/commits` entry.
- `src/lib.rs` — `pub mod project;` and `pub mod cost;`.

## 10. Explicitly out of scope for this pass (do not build these — named so they aren't silently forgotten)

1. **True parallel task execution.** The single shared checkpoint shadow-repo (section 2) and the single shared working tree make concurrent workers a real concurrency-safety problem: checkpoint races, overlapping file writes, interleaved live output. This pass stays sequential, exactly like `/swarm` today — "long-running" is satisfied by a project's tasks running one after another for as long as the spec needs, not by concurrency. A safe parallel-execution design (likely: per-worker git worktrees, or a task dependency graph with non-overlapping file ownership) is real, separate follow-up work.
2. **Cross-process live attach.** Checking on a project that's mid-run *from a different terminal/process* needs a lock+IPC mechanism or a background daemon — a materially different architecture, and closer to the OpenClaw always-on-agent shape this feature is explicitly trying to avoid. This pass's "browse anytime" means reading the persisted, saved-after-every-task state file (accurate as of the last finished task), not attaching to a live process mid-turn.
3. **Editing an accepted spec's tasks individually** after `/project new` finishes (e.g. "actually skip task 4"). v1: the only edit path is another `/project new` conversation producing a new project.
4. Hard mid-turn budget enforcement (stopping *during* a single expensive turn, not just between tasks). v1 checks budget between tasks only (section 8) — simpler, and tasks are bounded in size by the spec anyway.

## 11. Suggested build order

1. `src/cost.rs` + unit tests. No dependents yet, safe to land alone.
2. `swarm_agent` return-type change (section 5) + update its 3 call sites in `run_swarm`, ignoring the new token values for now. **Regression-test `/swarm <goal>` end-to-end before continuing** (section 12).
3. `src/project.rs`: data model, persistence functions, `slug_for`, `format_summary`, `PLANNER_SYSTEM`, `parse_spec_ready`. Unit tests for slug generation, JSON round-trip, spec-ready parsing (mirror `swarm.rs`'s test style).
4. Extract `run_task_loop` out of `run_swarm` (section 6); `run_swarm` calls it, behavior unchanged. **Regression-test `/swarm <goal>` again.**
5. `/project new` conversational loop (section 7).
6. `/project run` (section 8), calling `run_task_loop` with a `Project`-backed callback.
7. `browser::BrowserMode::Projects` + `run_projects` (section 9).
8. Command wiring + docs (section 9's mechanical checklist).
9. Full live pty verification (section 12).

## 12. Verification

1. `cargo build --release && cargo clippy --release --all-targets` clean at every step in section 11, not just at the end.
2. `cargo test --release` — new unit tests for `cost.rs` and `project.rs` pass alongside everything existing.
3. **Regression checks after steps 2 and 4 of the build order**: run a plain `/swarm <goal>` end-to-end (a small real goal in a scratch repo) and confirm it plans, works, checks, and finishes exactly as it did before this feature existed — same `/swarm-status` output shape, same session log event names. This is the step most likely to introduce a subtle bug, since it touches proven, working code.
4. Live end-to-end pty test of the new feature, same technique already used for `/commits` and the `!` shell-escape prompt-color feature earlier in this project's history (a Python `pty.openpty()` + `os.fork()` + `os.execvp("junebug", ...)` harness that writes keystrokes and reads back rendered terminal output — see git history / ask the user for the exact script if not obviously reconstructable): `/project new` → scripted back-and-forth → confirm `SPEC: READY` triggers the confirmation prompt → accept → confirm the project file exists in `.junebug/projects/` → `/project run <id>` → confirm task-by-task progress, checkpointing, and `cost` populate on disk → open `/project` and confirm the browser lists it with correct status/cost text → confirm Enter on a `Draft`/`Paused` project actually resumes execution → confirm `/project status <id>` matches the browser's preview text.
5. Manually verify the budget-pause path: set a very low `budget_usd` on a test project, run it, confirm it pauses after the first task that pushes the ledger over budget rather than continuing, and confirm `/project run <id>` again resumes correctly.
