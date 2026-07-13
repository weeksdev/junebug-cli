# Codex handoff

**Updated 2026-07-13 15:52 CDT.** This replaces the old ROUTING_HANDOFF.md
(routing shipped; deleted). Read [HANDOFF.md](HANDOFF.md) for full project
state.

## Nothing is in flight

The `/keys` feature that was mid-verification at 15:47 **finished verifying
and is committed**. The live pty test (`scripts/keys_smoke.exp`, isolated
HOME, no provider env vars) passed end-to-end: no-model startup banner and
guards, `/keys` provider menu, masked key entry, `model ready:
deepseek-v4-flash (deepseek)` brought up in-place, `/status` correct,
credentials file written 0600 with the key, and a grep proving the key never
entered any session log.

Current state on `main`:

- `d9a67a1` — routing, checkpoints + `/rewind`, MAX_TURNS 1000, sticky
  model, file diffs, model swarms.
- the commit after it — `/keys` command, no-model startup
  (`Option<OpenAiCompatibleProvider>` through `run()`/`repl()`), refreshed
  editor slash-completion list, README/HANDOFF sync, the smoke script.

Verification gate (`cargo fmt && cargo test && cargo clippy --all-targets
-- -D warnings && git diff --check`) was green at commit time: 61 unit +
4 integration tests.

## Where to pick up next

Highest-value candidates, roughly in order (see HANDOFF.md "Known gaps"):

1. OS-level sandboxing for `run_command` (seatbelt/landlock) — turns the
   approval flow into a hard boundary; the research notes in HANDOFF still
   apply.
2. Esc-interrupt inside `/swarm` (currently Ctrl-C kills the process;
   checkpoints protect the workspace) and headless `exec` swarm support.
3. Command timeouts + output truncation before buffering in
   `Workspace::run_command`.
4. Line-editor long-line wrap handling (cursor math assumes one row).
5. febo-api (private repo `~/repos/febo-api`, github.com/weeksdev/febo-api):
   auth tokens + rate limits, opt-in outcome telemetry.

Conventions that bite: run the gate before/after every slice; update
README.md and HANDOFF.md with any user-facing change; keep keys out of
patches, sessions, and commits; commit directly on main (Andrew's current
preference).
