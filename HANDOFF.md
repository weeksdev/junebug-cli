# Febo CLI handoff

## Purpose and current state

This repository is an early Rust implementation of **Febo CLI**, a local-first agentic coding CLI. The full product roadmap is in [PLAN.md](PLAN.md). Do not represent the project as feature-complete: the core agent loop is working, but major v0.2–v1 capabilities remain unimplemented.

The project lives at <https://github.com/weeksdev/febo_cli> (main branch; tags `v*` trigger the release workflow). It was renamed from "Junie" to "Febo" before first publish because JetBrains ships an AI agent named Junie. Avoid broad rewrites until you understand the safety boundaries below.

## What works today

- Real streamed REST calls through OpenAI-compatible Chat Completions endpoints:
  - `openrouter` (default), via `OPENROUTER_API_KEY`
  - `openai`, via `OPENAI_API_KEY`
  - `deepseek`, via `DEEPSEEK_API_KEY`
- API keys can be supplied through environment variables or a local ignored `.env` file.
- Model-driven multi-turn tool loop with streamed text, tool calls, tool results, token usage, cancellation-by-turn-limit, JSONL headless output, and local JSONL sessions.
- An interactive REPL (`febo` with no prompt and a terminal attached) with a Claude Code-style UI:
  - agent turns run on a worker thread (`std::thread::scope`) while the main thread renders `TurnEvent`s from an mpsc channel: streamed Markdown (`src/markdown.rs`, line-buffered ANSI styling), a braille spinner with elapsed seconds, `⏺ tool(args)` / `⎿ result` activity lines, and approval prompts (raw mode is dropped for the y/N read, then restored);
  - **Esc (or Ctrl-C) interrupts a running turn**: a shared `AtomicBool` is checked per SSE line in the provider (partial tool calls are discarded, a `[response interrupted by user]` marker is appended to the recorded assistant text) and before each tool execution (pending calls get `ERROR: interrupted by user` results so tool_call pairing stays valid);
  - a raw-mode line editor (`src/editor.rs`) with completion menus: `/` completes slash commands, `@` searches workspace files (name-prefix ranked, hidden/target/node_modules skipped, 2000-file/8-depth caps); Tab/Enter accept, ↑/↓ navigate menu or input history; Ctrl-A/E/U/K editing; falls back to plain `read_line` without a terminal;
  - `@path` mentions attach the file contents (via the policy-guarded `Workspace::read_file`) to the outgoing user message in `<attached-file>` tags;
  - slash commands: `/help`, `/status`, `/diff`, `/exit`, `/model [NAME]` (lists models from the provider's standard `GET /models` endpoint, switches with exact/unique-substring matching, best-effort), `/compact` (asks the model for a summary, then replaces in-memory history with system prompt + summary; the session log keeps raw history — plain `--resume` replays the original messages);
  - a failed turn (provider error, turn limit) reports the error and keeps the REPL alive. Without a terminal or with `--json`, an empty prompt is an argument error.
- `--resume-compact <session.jsonl>`: like `--resume` but runs the same summarization compaction before the first turn when the serialized history exceeds 4000 chars (skips with a note otherwise). Works in both REPL and single-shot modes.
- Headless `exec --json` now also emits `tool.call` events alongside `text.delta`/`tool.result`/`completed`. JSON event keys are serialized in alphabetical order.
- Workspace tools: `list_dir`, `read_file`, `search` (ripgrep), `write_file`, `run_command`, `git_status`, and `git_diff`.
- Safety controls:
  - default permission is now `read-only`;
  - `ask` prompts for writes and commands;
  - `workspace-write` permits only workspace writes; commands always prompt;
  - protected paths include `.env*`, `.git`, and `.febo` at any depth (component-based, so Windows separators and nested entries are covered);
  - path traversal is rejected, and symlinks are fully resolved: a symlink inside the workspace (including a dangling one) cannot read or create content outside it;
  - destructive/network-oriented shell commands are classified by a token-based matcher (whole command words plus shell separators, so `git add .` is not flagged while `rm<TAB>-rf` and `echo hi;rm -rf /` are; redirects to `/dev/null` and fd dups like `2>&1` are allowed, file redirects are flagged). **Flagged commands are no longer hard-blocked**: since every command requires explicit interactive approval, the classification adds a ⚠ warning to the approval prompt instead — a human seeing the exact command text is the gate, and headless runs can never approve;
  - all tool dispatch, including MCP tools, is routed through a dedicated `PolicyEngine` (see below) before any side effect runs; the MCP approval prompt shows the exact (truncated) arguments.
- `AGENTS.md` discovery, `.febo/hooks.json` hooks behind `--enable-hooks`, and local stdio MCP support behind `--enable-mcp`.
- Session continuation with `--resume <session.jsonl>`, and context character-budget compaction with `--max-context-chars`.
- Plan mode has a hard read-only permission guard and filters out write/command tools.
- CI exists at [.github/workflows/ci.yml](.github/workflows/ci.yml) for formatting, tests, and Clippy across macOS, Linux, and Windows.

## Verified evidence

The most recent local gate passed after the `PolicyEngine`/agent-loop extraction:

```sh
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings
git diff --check
```

At that point there were 35 Rust unit tests plus 3 integration tests in `tests/policy_integration.rs`. Live DeepSeek smoke tests (driven with `expect` over a pty — note `script(1)` is unreliable for this) verified in disposable temp workspaces: streamed Markdown rendering with tool activity lines and approvals; Esc mid-stream interrupting the turn with the follow-up message continuing from the partial reply; `/model` listing live models; `/compact` (11 → 3 messages); `--resume-compact` on a 28 KB session (follow-up turn cost 1371 input tokens instead of ~7k) and its small-history skip path; slash-command and `@file` completion menus including attachment expansion the model could answer from; `list_dir(.)`; and an approved `git log --oneline -5 2>/dev/null || echo none`. Do not use or print credentials from `.env`; it is ignored.

A full-architecture review after the `PolicyEngine` extraction found and fixed these defects (each has a regression test):

1. **Symlink read/write escape.** `checked_path` canonicalized only the parent, so a symlink file inside the workspace could read content outside it, and a dangling symlink could create a file outside it via `fs::write` following the link. The target itself is now fully resolved when it exists (via `symlink_metadata` + `canonicalize`). A follow-up fix: for `list_dir(".")` the parent-ancestor fallback normalized past the workspace root and rejected it; existing targets now short-circuit on their own canonical containment and the ancestor walk applies only to not-yet-existing targets.
2. **Compaction could orphan `tool` messages.** Cutting between an assistant `tool_calls` message and its `tool` results produced requests OpenAI-compatible APIs reject with 400. `context::compact` now widens past the budget until the boundary is valid.
3. **30-second stream death.** `reqwest::blocking::Client::new()` defaults to a 30-second whole-request timeout, killing any streamed turn longer than that. The provider now builds a client with a 30 s connect timeout and 10 min per-turn cap.
4. **Dangerous-command matcher was substring-based.** It blocked `git add .` (contains `dd `) and `ls sudoku/` (contains `sudo`) while missing `rm<TAB>-rf` and `;rm` chains. Replaced with token-based matching.
5. **Invalid JSONL output.** Hand-rolled `json_escape` missed control characters (`\t`, `\r`); exec-mode events are now serialized with `serde_json`. Note the event key order changed (alphabetical); the schema fields are unchanged.
6. **Protected paths were `/`-string-matched.** `.git\config` on Windows and nested `.env`/`.git` entries bypassed protection; matching is now per path component at any depth.
7. **Docs/help drift.** README and `--help` claimed the default permission is `ask`; it is `read-only`. README claimed MCP/hooks are unavailable; both exist behind `--enable-mcp`/`--enable-hooks`. `--plan` was missing from help.
8. Smaller: `write_file` can now create nested parent directories inside the workspace; plan mode no longer connects/offers MCP servers (their calls were all policy-denied anyway).

## Important implementation notes

| Area | Main files | Notes |
| --- | --- | --- |
| CLI entrypoint | [src/main.rs](src/main.rs) | Arg parsing, provider/session/hooks/MCP wiring, and the interactive stdin approval prompt. It delegates tool dispatch and the turn loop to `agent`; still growing and a candidate for further splitting (e.g. arg parsing into its own module). |
| Agent loop/tool gateway | [src/agent.rs](src/agent.rs) | `run_loop` drives the model-turn loop; `execute_tool` classifies each call's `ToolRisk` (including MCP tools, now always `Execute`) and enforces the `PolicyEngine` decision before dispatch. Approval I/O is injected via an `approve: &mut dyn FnMut(&str) -> bool` closure so the loop is testable without a terminal. |
| Policy engine | [src/policy.rs](src/policy.rs) | Deterministic `PolicyEngine::evaluate(risk) -> Decision {Allow, Deny, Ask}` from permission mode + plan-mode only — no model/tool-argument input. Plan mode is now a hard guard enforced here, not only via tool-list filtering. Unit-tested exhaustively across the mode × risk matrix. |
| REST provider | [src/provider.rs](src/provider.rs) | Only OpenAI-compatible Chat Completions SSE is live. Anthropic Messages and OpenAI Responses are not implemented. |
| Workspace tools | [src/tool.rs](src/tool.rs) | Filesystem/command/git primitives and `ToolRisk`/`BUILTIN_TOOLS` classification; the caller (`agent::execute_tool`) enforces policy before invoking anything here. |
| Sessions | [src/session.rs](src/session.rs) | JSONL structured messages plus audit events; resumes supplied session paths. |
| Instructions/context | [src/instructions.rs](src/instructions.rs), [src/context.rs](src/context.rs) | Context compaction is character-based, not token-based; no summarizer yet. |
| Hooks/MCP | [src/hooks.rs](src/hooks.rs), [src/mcp.rs](src/mcp.rs) | MCP is stdio-only and minimal; no HTTP/OAuth, timeout, per-tool approval, or robust protocol tests. MCP tool calls are now gated through the same `PolicyEngine` as built-ins (previously they bypassed permission checks entirely). |

## Known gaps and risks

These are the highest-priority items, in order:

1. **Complete v0.1 correctness and UX.** The interactive REPL now has streamed Markdown, Esc/Ctrl-C turn interruption, completion menus, `/model`, and `/compact`; still missing are `--cwd`, `/undo` with diff preview, precise patch/apply-patch support, command timeouts, output truncation before buffering, tool approval audit records, secret redaction, provider/runtime integration fixtures, long-line wrap handling in the line editor (cursor math assumes one row), and completion of model names after `/model `. Note the editor and turn UI require a real terminal; everything falls back to plain stdio without one.
2. **Documentation drift is fixed as of the architecture review** — [README.md](README.md) now documents the real default permission (`read-only`), `--plan`, `--resume`, hooks, and MCP. Keep it in sync. [PLAN.md](PLAN.md) still lists items as deferred because it is the original roadmap, not status tracking.
3. **Refactor into core contracts.** The plan calls for `ModelProvider`, `Tool`, `PolicyEngine`, `SessionStore`, and `PromptBuilder`. `ModelProvider`, `SessionStore` (as `session.rs`), and `PolicyEngine` (as `src/policy.rs`, with dispatch enforcement in `src/agent.rs`) now exist as dedicated modules; `Tool` and `PromptBuilder` are still informal (tool schemas are inline JSON in `main.rs`; there is no dedicated prompt builder beyond string formatting in `main.rs`).
4. **Provider compatibility.** Add and test Anthropic Messages and OpenAI Responses adapters, model capability discovery, retries/backoff, rate-limit handling, custom base URLs, and structured error contracts.
5. **Harden MCP and hooks.** MCP needs server timeout/restart controls, protocol/version tests, tool-level approvals, config provenance/pinning, and remote Streamable HTTP/OAuth/PKCE. Hooks need before/after tool events and allow/deny/observe semantics.
6. **Implement v0.3/v1.** Skills, agent profiles/subagents, worktree isolation, plugin manifests/integrity, local app-server JSON-RPC, policy profiles, packaging/signing, SBOM, docs, and release process are all absent.

## Safe continuation workflow

1. Read [PLAN.md](PLAN.md) completely and treat it as the source of product requirements.
2. Run the verification gate above before and after each meaningful slice.
3. Keep API keys out of patches, output, sessions, and commits. `.env` is intentionally ignored.
4. For every new executable capability, make the default deny/ask decision explicit and add a test proving no unapproved action runs.
5. Prefer a small vertical slice with unit + integration coverage over untested scaffolding.
6. When changing user-facing behavior, update [README.md](README.md) and this document.

## Suggested next task

The dedicated `PolicyEngine`/`agent` module described above is now in place, along with fixture-provider integration tests in `tests/policy_integration.rs`. Good next slices, in priority order:

1. Extract tool JSON Schemas out of the inline `json!` literals in `main.rs::tool_schemas` into a `Tool` contract (name, schema, `ToolRisk`) colocated with `tool.rs`'s `BUILTIN_TOOLS`, so the schema and the risk classification cannot drift apart as tools are added.
2. Build `apply_patch`/precise-patch support and a diff preview on top of the now-centralized `PolicyEngine::evaluate` gate — the write path no longer needs touching for this, only a new tool plus schema/risk entry.
3. Add command timeouts and output truncation *before* buffering in `Workspace::run_command` (currently buffers full output before checking the 1 MiB cap).
4. Add tool approval audit records (who/what/when a `Decision::Ask` was approved or denied) to `SessionWriter`, now that all approvals funnel through one place (`agent::execute_tool`).

