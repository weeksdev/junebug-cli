# Routing integration handoff

**Written 2026-07-12 16:47 CDT.** The routing API service is done; this document
hands off the `febo_cli` integration work. Read [HANDOFF.md](HANDOFF.md) first
for general project state and the verification gate.

## Product decisions already made (do not relitigate)

- **In-loop model switching is the feature.** Routing once per user prompt was
  explicitly rejected — the savings come from cheap models doing exploration/
  verification and strong models doing planning/repair *within one task*. The
  safe switch boundary is after a complete tool-call batch (assistant
  `tool_calls` message + all matching `tool` results appended), never between
  them.
- **Routing is strictly opt-in.** Default behavior is exactly today's: one
  provider, one model, chosen by the user. Opted-out users must see zero
  change. Opt-in via config (`routing.mode: "auto"`), `/model auto` in the
  REPL, or `--model auto` on the CLI. `/model NAME` or `/model provider:NAME`
  pins a model and disables routing; `/model auto` re-enables it.
- **It must always be obvious which model is running.** Requirements:
  - the status footer under the prompt shows the *current* provider/model;
  - every route change prints a one-line notice with the reason, e.g.
    `↳ gpt-5.4 (openai) — repeated tool failures during repair`;
  - tool activity while routed should make the active model visible (e.g.
    include the model in the turn-completion line `✓ 12s · tokens …` or on the
    `⏺ tool(...)` lines — pick one, keep it subtle but unambiguous);
  - `/status` shows routing mode, current route, band, and switches this task;
  - `exec --json` emits `route.selected` / `route.changed` events.
- **Privacy: derived signals by default, prompt opt-in** (`routing.send_prompt:
  true`). Never send code, filenames, diffs, or tool output to the routing API.
- The CLI stays open source; the routing service is a separate private repo.

## What already exists: febo-api (separate private repo)

`~/repos/febo-api` — axum HTTP service, committed locally (91e888b), not on
GitHub. `cargo run` binds `127.0.0.1:8791` (`FEBO_API_ADDR` overrides). 19 unit
tests; fmt/clippy clean; verified live with curl. Its README documents the full
contract with examples. Summary:

- `POST /v1/route` takes `{task, execution, routes, preferences}` and returns
  `{route, band, score, confidence, switch, reasons, recheck_after_turns}`.
  `422 {"error": …}` when `routes` is empty.
- Bands: `simple < standard < complex < critical`. The user maps each band to
  `{provider, model}`; unmapped bands escalate to the nearest mapped higher
  band, else fall back to the highest mapped lower band.
- `task.signals`: prompt_chars, attached_file_count, attached_content_chars,
  conversation_message_count, plan_mode, languages, and boolean `indicators`
  (security/concurrency/migration/architecture/refactor/trivial). The CLI must
  derive indicators locally (keyword match on the prompt) so the prompt itself
  need not be sent. `task.prompt` (optional, opt-in): the server re-derives
  indicators from it and ORs them with the client's.
- `execution`: turn_index, turns_remaining, phase
  (understand/explore/plan/implement/verify/repair/review), consecutive_tool_
  failures, tests_failing, provider_error, previous_band, turns_on_route,
  switches_so_far. All optional/defaulted — an empty `execution` means
  "initial classification".
- `preferences`: thresholds {simple_max 0.25, standard_max 0.55, complex_max
  0.78}, min_band (quality floor), max_band (cost cap, wins), minimum_turns_
  on_route (2), max_switches (5).
- Server-side behavior the CLI can rely on: high-risk indicators and ≥2
  consecutive tool failures floor the band at `complex`; hysteresis holds the
  previous band unless residency is met, the switch budget remains, and
  downgrades clear a 0.05 margin; emergencies (≥2 failures, provider_error)
  bypass residency/budget. Every decision carries human-readable `reasons`.

## CLI integration plan

### 1. Prerequisite refactor: explicit model per call

`OpenAiCompatibleProvider` (src/provider.rs) is stateful: one kind + one model.
Change the `ModelProvider` trait so the model is a per-call argument:

```rust
fn stream_turn(&self, model: &str, messages: &[Value], tools: &[Value],
               cancel: &AtomicBool) -> Result<ModelTurn, String>;
```

Keep `model()`/`set_model()` on `OpenAiCompatibleProvider` as the *pinned*
model for non-routing use. Add a `ProviderRegistry` that lazily constructs one
`OpenAiCompatibleProvider` per `ProviderKind` with a credential
(`available_providers()` already exists). The fixture provider in
`tests/policy_integration.rs` implements `ModelProvider` and must be updated.

### 2. Config: `~/.febo/config.json` (+ workspace `.febo/config.json` override)

```json
{
  "routing": {
    "mode": "off",                          // "off" (default) | "auto"
    "api_url": "http://127.0.0.1:8791",
    "send_prompt": false,
    "routes": {
      "simple":   {"provider": "deepseek",   "model": "deepseek-chat"},
      "standard": {"provider": "openrouter", "model": "qwen/qwen3-coder"},
      "complex":  {"provider": "openai",     "model": "gpt-5.4"},
      "critical": {"provider": "openai",     "model": "gpt-5.6"}
    },
    "max_band": null,
    "min_band": null
  }
}
```

New module `src/config.rs`, plain serde. Only include routes whose provider
has a credential when building the API request; if none remain, routing
degrades to pinned-model mode with a printed notice.

### 3. `src/router.rs`

- Serde types mirroring the wire contract (keep them CLI-local; do not share a
  crate with febo-api).
- `derive_indicators(prompt)` — same keyword heuristic as the server
  (security/concurrency/migration/architecture/refactor/trivial).
- `RouterClient` — reqwest::blocking POST with a short timeout (~750 ms
  connect, ~2 s total). On any error: fall back to `LocalRouter` (a small
  rule-based copy: thresholds → band, failure floor, residency) and print a
  dimmed one-time notice `routing API unreachable — using local rules`.
- `RoutingState` tracked by the loop: current band/route, turns_on_route,
  switches_so_far, consecutive_tool_failures (reset on a successful tool
  result; count results starting with `ERROR:`), tests_failing (heuristic:
  last run_command result contained test failures — optional v1), phase
  (optional v1: derive from last tool — search/list/read ⇒ explore,
  write_file ⇒ implement, run_command ⇒ verify; absent is fine, the API
  defaults it).

### 4. Loop changes (src/agent.rs)

`run_loop` currently takes `provider: &dyn ModelProvider` (agent.rs:50). For
routing it needs a model source consulted once per model turn, at the top of
the `for` loop — which is exactly the safe boundary, since tool results from
the previous iteration are already appended (agent.rs:107-122). Suggested
shape, keeping the fixture tests simple:

```rust
pub trait ModelSource {
    /// Called before each model turn; returns the provider, model, and an
    /// optional user-facing notice when the route changed.
    fn next(&mut self, state: &TurnState) -> Result<Selection<'_>, String>;
}
```

with a `PinnedModel` implementation (wraps one provider + model; never
notices) and a `RoutedModel` implementation (registry + RouterClient +
RoutingState). `TurnState` carries what the router needs (turn_index,
consecutive failures, etc.). Add `on_route_changed(&decision)` to
`TurnObserver` (agent.rs:31) with the REPL printing the `↳ …` line, the plain
observer emitting the JSONL event, and record `route_selected` /
`route_changed` in the session (`SessionWriter::record`).

### 5. UI/UX wiring (src/main.rs)

- `status_footer` (main.rs:633): show the *current* route's model, plus
  `auto` marker when routing, e.g. `auto:gpt-5.4 · ask · …`.
- `banner` (main.rs:1061): show `routing: auto (4 bands)` or the pinned model.
- `/model` command (main.rs:718): add `auto` argument; pinning any model sets
  mode to off for the session and prints `routing disabled — pinned to X`.
- `/status`: add routing mode, current band/model, switch count.
- Turn completion line (main.rs:1007): include the model that finished the
  turn when routing is on.
- `exec --json`: `route.selected` on first selection, `route.changed` on
  switches. Keys alphabetical (serde_json object serialization already does
  this — see HANDOFF.md defect 5).

### 6. Tests

- Unit: config parsing (defaults, missing file), indicator derivation, local
  fallback router decisions, RoutingState failure counting.
- Integration (tests/policy_integration.rs style): a fixture `ModelSource`
  that switches between two fixture providers mid-task and asserts (a) the
  tool_call/tool_result pairing stays valid across the switch, (b) observer
  saw the route-change notice, (c) session recorded `route_changed`.
- Do NOT add tests that hit the live API; the client is exercised via the
  fallback path plus (optionally) a `std::net::TcpListener` stub returning a
  canned response.

### Sequencing

1. Refactor `stream_turn(model, …)` + registry + fix fixture tests (no
   behavior change; gate must pass).
2. Config + router module + local fallback (unit-tested, unused).
3. `ModelSource` in run_loop + observer/session events + REPL/JSON wiring.
4. Live smoke test: run febo-api locally, `/model auto` in a disposable
   workspace, verify visible switch notices; also verify `/model NAME` still
   pins with zero routing traffic (no HTTP calls when mode=off — check with
   the API server stopped).

Gate before and after every slice (from HANDOFF.md): `cargo fmt && cargo test
&& cargo clippy --all-targets -- -D warnings && git diff --check`. Update
README.md and HANDOFF.md when the user-facing behavior lands.
