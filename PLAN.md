# Febo CLI — product and implementation plan

## 1. Product definition

Febo CLI is an open-source, local-first coding agent that runs in a terminal and connects to Febo-hosted models. It must work with both an **OpenAI-compatible** and an **Anthropic-compatible** REST API so Febo model serving can evolve independently from the CLI. It should be useful as an interactive developer companion and dependable in CI/automation.

### Product principles

1. **Safe by default.** A model never gets implicit authority to modify files, run a command, access the network, or invoke an external tool.
2. **Local-first and inspectable.** Keep source, session data, configuration, approvals, diffs, and tool transcripts on the developer's machine by default.
3. **Provider-neutral core.** Normalize model events and tool calls behind an adapter; do not leak provider-specific message schemas into agent logic.
4. **Excellent terminal ergonomics.** Streaming output, readable diffs, keyboard controls, resumable sessions, and a headless JSONL interface are first-class.
5. **Extensible without blind trust.** MCP, project instructions, hooks, skills, and plugins are explicit trust boundaries and receive least privilege.

## 2. Research findings and resulting requirements

Leading coding CLIs converge on a few categories:

| Category | Requirement for Febo CLI |
| --- | --- |
| Interaction | REPL, one-shot/headless prompt, plan mode, file/image input, model selection, session resume/fork, searchable history, streaming and cancellation. |
| Agent loop | Model-driven tool calls; structured tool results; token/context accounting; compaction/summarization; recovery from transient API/tool failures. |
| Workspace tools | Fast search, read files in bounded chunks, apply precise patches, unified diffs, shell/PTY execution, git status/diff/log, and optional LSP navigation. |
| Safety | Separate read/write/command/network permissions, workspace and additional-directory allowlists, command/path policies, approval prompts, sandboxing, secret handling, audit log, and dry-run/plan mode. |
| Customization | Hierarchical project instructions (`AGENTS.md`), reusable skills, slash commands, hooks, custom agent profiles/subagents, and versioned configuration. |
| Interoperability | MCP client (stdio first, then Streamable HTTP + OAuth); OpenAI-style and Anthropic-style API adapters; stable JSONL/JSON-RPC integration surface. |
| Reliability | Resumable append-only sessions, deterministic tool event recording, cancellation, bounded retries, rate-limit handling, telemetry that is opt-in, and an offline test fixture provider. |

These are not speculative features: Claude Code documents instructions/skills/subagents/hooks/MCP as its extension model; Copilot CLI offers plan/autopilot modes, JSONL prompt mode, LSP, persistent sessions, hooks, MCP, plugins and subagents; Codex exposes non-interactive execution and a bidirectional JSON-RPC app server. Sources are listed in [Appendix A](#appendix-a--research-sources).

## 3. Scope and release sequence

### v0.1 — trustworthy single-agent CLI (MVP)

**Goal:** A developer can ask Febo to inspect a repository, make a small change, approve every material action, review the diff, and resume the conversation later.

- CLI: `febo`, `febo "prompt"`, `febo exec "prompt"`, `--model`, `--provider`, `--cwd`, `--resume`, `--json`, `--version`.
- Full-screen terminal UI with streaming markdown/code, activity timeline, approval dialog, diff view, Ctrl-C stop, and slash commands: `/help`, `/model`, `/status`, `/diff`, `/undo`, `/compact`, `/exit`.
- Agent runtime: a bounded iterative tool loop; max-turn/max-token limits; cancellation; tool-result truncation; retries with exponential backoff; visible failure state.
- Built-in tools: `list_dir`, `search` (ripgrep), `read_file`, `apply_patch`, `write_file`, `run_command`, `git_status`, `git_diff`.
- Permission modes: `read-only` (default), `ask` (ask for writes/commands/network), and `workspace-write` (pre-approve writes only under the workspace). Never pre-approve destructive shell commands or network access in v0.1.
- Shell commands execute with cwd, timeout, output cap, sanitized environment, and clear command preview. File writes must be patch-based and presented as a diff before or immediately after execution.
- Load `AGENTS.md` from repository root and ancestor directories; show which instruction files were loaded and their precedence. Include `--no-project-instructions`.
- Local session store (SQLite or JSONL): transcript, tool arguments/results, approval decisions, model/provider metadata, timestamps, usage, summary/compaction artifacts. Redact configured secret patterns before persistence.
- Headless `exec` output is JSON Lines with a documented event schema, non-zero exit status on failure/cancellation, and no TUI control characters.
- Tests: provider contract tests, tool safety tests, approval snapshots, terminal-free runtime integration tests with fake model responses, and cross-platform smoke tests for macOS/Linux/Windows.

**Explicitly deferred:** MCP, arbitrary hooks/plugins, autonomous mode, subagents, remote control, cloud session sync, and automatic code edits without a user approval.

### v0.2 — usable daily-driver integrations

**Goal:** Expand context and integrations without weakening the trust model.

- OpenAI-compatible Responses/Chat Completions adapter and Anthropic Messages adapter, both supporting SSE streaming and normalized tool calls. Publish exact supported request/event subsets and conformance fixtures.
- Authentication profiles: API key from environment/keychain/config, custom base URL, organization/project headers where applicable, redacted diagnostics, health check, and `febo auth login/logout/status`.
- Attach local images and files, `@path` mentions, clipboard/piped stdin, ignore rules, binary/large-file protection, and token-aware repository context selection.
- MCP client: local stdio servers first; per-server enablement and tool approvals; namespaced tool names; explicit startup/trust prompt; tool list pinning/audit record. Add Streamable HTTP and OAuth/PKCE only after the local client is robust.
- Hooks with allow/deny/observe decisions at lifecycle points (`session_start`, `before_tool`, `after_tool`, `session_end`). Repository hooks remain disabled until a user trusts that workspace.
- Optional LSP adapter for definition, references, diagnostics, and symbol rename. It augments, never replaces, safe filesystem edits.
- Git workflows: branch awareness, commit/message proposal, test/lint command recommendations, and change summary. Do not push or create a PR without explicit approval.

### v0.3 — scalable agent workflows

**Goal:** Make complex tasks manageable while maintaining deterministic authority and context boundaries.

- Plan mode: research and write a structured plan, identify files/tests/risk, then require explicit transition to execute mode.
- Skills: portable directories containing `SKILL.md`, optional scripts/resources, metadata, provenance, version, and granted capabilities. Discover user and workspace skills; require consent before executing bundled code.
- Agent profiles and subagents: isolated context/session, explicit tool/permission budget, concurrency/cost limits, parent-only write authority by default, and merged findings with provenance.
- Worktree isolation for implementation agents, optional reviewer/tester profiles, handoff summaries, and conflict detection before applying changes.
- Context engine: prioritised instructions, working set, git diff, tool outputs, compressed summaries, token budget display, and manual `/context` inspection.
- Review/verification loop that runs selected tests, reads failures, and presents a final implementation report (changed files, test results, remaining risks).

### v1.0 — stable open platform

**Goal:** Commit to a stable, secure developer and integration experience.

- Stable config, plugin, skill, JSONL event, and provider-adapter interfaces with semver and migration policy.
- Plugin marketplace/install/update/uninstall, signed manifests or integrity hashes, capability declarations, provenance display, and revocation/disable controls.
- Optional local JSON-RPC app-server protocol for IDEs and alternate UIs; versioned protocol schema, bidirectional events, and compatibility tests.
- Policy profiles suitable for teams: protected paths, deny rules, command/network allowlists, managed configuration precedence, exportable audit logs, and documented admin controls.
- Accessibility, Windows parity, secure auto-update/release signing, crash reporting only with opt-in, SBOM, vulnerability response process, and reproducible builds where practical.

## 4. Technical architecture

Keep the architecture layered so models, interfaces, and tools can change independently:

```
TUI / one-shot CLI / JSONL exec / future app server
                    │
          Session + command dispatcher
                    │
  Agent runtime ── Context builder ── Policy/approval engine
         │                │                    │
  Provider adapter    Session store       Tool gateway
 (OpenAI/Claude)                            │
                                      built-ins / MCP / LSP
```

### Core contracts

- **`ModelProvider`**: capability discovery, `stream_turn(request)`, cancellation, normalized `text_delta`, `reasoning_delta` (if offered), `tool_call`, `usage`, `error`, and `completed` events.
- **`Tool`**: machine-readable JSON Schema input/output, risk metadata (`read`, `write`, `execute`, `network`, `external`), validation, timeout/output budgets, and a redaction policy.
- **`PolicyEngine`**: produces `allow`, `deny`, or `ask` from workspace trust, tool risk, target paths/command/network host, CLI mode, and remembered approval scope. It must be deterministic and independently testable.
- **`SessionStore`**: append-only events plus derived state; session IDs are provider-agnostic. A session can be resumed under a different provider only with an explicit warning.
- **`PromptBuilder`**: stable system baseline + safety/policy state + selected instructions + task/session summary + recent events + tool schemas. It owns token budgeting and truncation, not individual tools.

### API compatibility strategy

Febo servers should ideally expose both front doors:

1. **OpenAI-compatible:** `POST /v1/responses` (preferred) and, if needed for ecosystem compatibility, `POST /v1/chat/completions`; `GET /v1/models`; Bearer auth; SSE streaming; JSON Schema function tools.
2. **Anthropic-compatible:** `POST /v1/messages`; `GET /v1/models` if Febo chooses to offer it; `x-api-key` and `anthropic-version` handling; SSE streaming; `tool_use` / `tool_result` blocks.

The CLI's canonical internal model must be richer than either API, then adapters translate at the edge. Start with the exact subset Febo models need for multi-turn text and tool calling. Add vision, prompt caching, structured output, and extended thinking as negotiated capabilities—not accidental assumptions based on a model name.

Define a capability endpoint or model metadata fields early: context-window, max-output, streaming, tools, parallel tool calls, images, structured output, reasoning visibility, and prompt caching. The UI must gracefully hide unsupported capabilities.

## 5. Security and trust requirements

- Treat model output, repository content, tool output, and MCP content as untrusted instructions. Never let any of them alter policy or system instructions.
- Enforce canonical paths; block traversal and symlink escape; distinguish workspace reads from additional approved directories; protect `.git`, credentials, SSH keys, and OS configuration by default.
- Parse shell commands only as far as needed for policy; do not infer safety from a natural-language explanation. Show the exact command, cwd, environment changes, and network destination before approval.
- Isolate command execution using an OS-appropriate sandbox where available. The safe fallback is a strict permission prompt, not a silently unsandboxed auto mode.
- Prevent secret disclosure in UI, logs, tool payloads, and model context via configurable detectors plus explicit protected-path rules. Detection is defense in depth, not a reason to scan arbitrary private files.
- MCP servers and repository-supplied hooks/plugins can execute arbitrary local code. Require first-use trust, keep them off in non-interactive runs unless explicitly enabled, and record their version/configuration in the audit trail.
- Store credentials in the OS keychain when possible; avoid plaintext config. Audit logs must be local by default and support redaction/export.

## 6. Suggested delivery milestones

| Milestone | Exit criteria |
| --- | --- |
| M0: foundation | Repository, language/toolchain decision, formatter/linter/test runner, release CI, architecture decision records, fake Febo provider, and threat model are in place. |
| M1: read-only agent | Interactive and headless prompts stream from fake/real provider; safe read/search tools work; `AGENTS.md`, session persistence, cancellation, and JSONL contracts are tested. |
| M2: guarded edits | Patch/write and shell tools use approval engine; diffs and undo work; integration fixtures prove no unapproved write/command executes. |
| M3: Febo API compatibility | Real OpenAI- and Anthropic-shaped servers pass adapter contract suite for streaming, errors, multi-turn tool use, usage, and cancellation. |
| M4: beta | Authentication profiles, operational diagnostics, package/install artifacts for three OSes, docs/tutorials, opt-in telemetry decision, and a public issue/security process. |
| M5: v1 | Stable compatibility and extension contracts, policy profiles, plugin integrity model, app-server decision, and backwards-compatible migration tooling. |

## 7. Decisions to make before implementation

1. **Implementation language:** Rust is a strong default for a cross-platform, single-binary CLI with process/sandbox control; TypeScript is faster for an ecosystem with mature terminal/MCP libraries. Make this an ADR after a one-day spike for TUI, PTY, and packaging—not a taste decision.
2. **License and governance:** choose an OSI license, contribution process, code of conduct, security policy, and whether extensions may use different licenses.
3. **Sandbox targets:** define the minimum supported behavior on macOS, Linux, and Windows; never market equivalent security guarantees without per-OS proof.
4. **Data posture:** decide whether Febo servers retain prompts/tool traces, whether self-hosting is supported, and the exact telemetry default. Document this before beta.
5. **API contract ownership:** version Febo API compatibility separately from the CLI. Publish fixtures early so server and client teams cannot drift.
6. **Initial target audience:** individual local developers vs. managed enterprise teams. The first shapes onboarding; the second materially expands policy/audit/SAML/egress scope.

## 8. Definition of done for v0.1

v0.1 is ready when a fresh developer can install one signed artifact, authenticate/configure a local Febo server, run `febo` in a Git repository, ask it to change code, see and approve each write/command, inspect the final diff and test output, resume the session, and use `febo exec --json` reliably in a script. The project must publish a threat model, permission documentation, API adapter contract tests, and a reproducible demo repository—not merely a feature list.

## Appendix A — research sources

- [Claude Code feature overview](https://code.claude.com/docs/en/features-overview) and [permissions](https://code.claude.com/docs/en/permissions): extension model and permission enforcement.
- [GitHub Copilot CLI overview](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli), [CLI reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference), and [customization comparison](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/comparing-cli-features): modes, JSONL execution, sessions, LSP, subagents, hooks, MCP, plugins.
- [OpenAI Codex CLI README](https://github.com/openai/codex/blob/main/codex-rs/README.md) and [app-server protocol README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md): programmatic execution and bidirectional JSON-RPC integration.
- [OpenAI Responses streaming reference](https://platform.openai.com/docs/api-reference/responses-streaming/response/refusal/delta?lang=curl): SSE and function-tool event model.
- [MCP authorization specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization): OAuth/PKCE and token-audience requirements for remote MCP.

