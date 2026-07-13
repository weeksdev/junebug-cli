# Febo CLI

Febo CLI is an open-source, local-first coding agent for Febo models.

This initial foundation intentionally defaults to read-only workspace access. It includes a provider-neutral event contract, safe read/search tools, and streaming REST support for OpenAI, OpenRouter, and DeepSeek.

Install on macOS (builds from source, installs to `~/.local/bin/febo`):

```sh
./install-macos.sh
```

Tagged versions (`v*`) publish prebuilt binaries for macOS (arm64/x86_64), Linux, and Windows on the [GitHub releases page](https://github.com/weeksdev/febo_cli/releases).

Save a provider API key once (stored in `~/.febo/credentials.env`, `0600`) and start chatting. When you omit `--provider` and `--model`, Febo defaults to the provider **and model** from your last session in this workspace (falling back to whichever key it can find and the provider's default model):

```sh
febo set --provider deepseek YOUR_API_KEY
```

Or just run `febo` with nothing configured: it starts in no-model mode and `/keys` sets a key from inside the REPL (arrow-key provider picker, hidden input) and brings a model up on the spot — no restart. Keys never appear in session logs and are never exposed to the model.

```sh
cargo run -- --help
OPENROUTER_API_KEY=... cargo run -- "Describe this project"
OPENAI_API_KEY=... cargo run -- --provider openai --model gpt-4.1-mini "Describe this project"
DEEPSEEK_API_KEY=... cargo run -- --provider deepseek --model deepseek-v4-flash "Describe this project"
```

OpenRouter is the default provider. Use `--model` to select a model supported by the provider. See [PLAN.md](PLAN.md) for the release roadmap.

Automatic in-task routing is strictly opt-in: use `--model auto`, `/model auto`, or set `routing.mode` to `"auto"` in `~/.febo/config.json` (a workspace `.febo/config.json` overrides it). Configure `routing.routes` with band entries containing `provider` and `model`. Only providers with credentials are offered. By default Febo sends the routing service derived task signals—not prompts, code, filenames, diffs, or tool output. Set `routing.send_prompt` to `true` only if you explicitly want the prompt sent. The default service URL is `http://127.0.0.1:8791`; if it is unavailable, Febo prints a notice and uses local routing rules.

The CLI uses standard provider environment variables and also reads an ignored `.env` file from the current workspace. Copy `.env.example` to `.env`; values are never printed or persisted in session logs.

## Current commands

- `febo` — interactive REPL with a Claude Code-style terminal UI: streamed Markdown rendering, a spinner with elapsed time, `⏺ tool(args)` / `⎿ result` activity lines, and **Esc to interrupt** a running turn (the partial reply stays in context; your next message continues from it).
  - Slash commands: `/help`, `/keys` (set or replace a provider API key, input hidden), `/model` (arrow-key pick from the provider's live model list), `/permissions` (arrow-key switch between read-only / ask / workspace-write / **yolo** mid-session), `/rewind` (restore workspace files to an earlier checkpoint), `/compact` (model-written summary replaces old history), `/status`, `/diff`, `/exit`. A dimmed status line under the prompt always shows the current model and access level.
  - Input intellisense: typing `/` opens a slash-command menu; typing `@` opens a workspace file search menu (↑/↓ select, Tab/Enter accept). Accepted `@path` mentions attach the file's contents to your message.
  - Line editing: arrows, Home/End, Ctrl-A/E/U/K, and ↑/↓ history.
- `febo [prompt]` — single prompt execution with approval prompts for writes and commands.
- `febo exec [--json] <prompt>` — script-friendly execution emitting JSON Lines events (`text.delta`, `tool.call`, `tool.result`, `route.selected`, `route.changed`, `completed`).
- `--permission read-only|ask|workspace-write|yolo` — defaults to `read-only`. `yolo` approves every write **and** command without asking (plan mode still forces read-only). Non-interactive runs require `workspace-write` or `yolo` for edits.
- `--plan` — hard read-only guard that also hides write/command tools from the model, regardless of `--permission`.
- `--resume [session.jsonl]` — continue a session; **with no path it lists past sessions to pick from**. `--resume-compact [session.jsonl]` additionally summarizes a large history first so resuming does not waste tokens. `--max-context-chars` bounds the request context with deterministic compaction.
- **Model swarms** — `/swarm-setup` assigns models to three roles: a **boss** that inspects the workspace, writes a constitution + task plan, rules on disputes, and reviews the result (use your strongest model — it never writes files); a **worker** that executes each task (cheap); and a **checker** that independently verifies every task with its own tools, never trusting the worker's report (cheap). `/swarm <goal>` runs the loop: failed checks send specific feedback back to the worker (up to 2 reworks), persistent disputes escalate to the boss — which can overrule the checker — and the boss closes with an honest final report. Every step is labeled with the role and model doing it, each swarm gets its own session log, and checkpoints are taken throughout so `/rewind` can undo a swarm. Requires `ask` or `workspace-write` permission (`yolo` avoids per-step prompts).
- **File-change diffs** — every file write shows what changed: colored `-`/`+` line hunks with context appear in the REPL activity stream under the tool line, in `ask`-mode approval prompts (so you see exactly what you're approving before answering y/N), and as `file.diff` events in `exec --json`. Diffs are display-only and are never sent to the model.
- **Workspace checkpoints** — before every prompt, file write, and command, Febo snapshots the workspace into a shadow Git repository under `~/.febo/checkpoints/` (your own repo is never touched; works in non-Git workspaces). `/rewind` restores files to any checkpoint, and the pre-restore state is checkpointed first so a rewind is always undoable. Snapshots respect your `.gitignore` and never capture `.env*`, `.febo/`, or build caches. Disable with `--no-checkpoints`.
- `--enable-hooks` — run explicitly trusted `.febo/hooks.json` lifecycle commands.
- `--enable-mcp` — connect local stdio MCP servers from `.febo/mcp.json`; every MCP tool call requires an interactive approval showing the arguments.
- `febo --version` / `febo --help`.

Sessions are recorded locally in `.febo/sessions/` as JSON Lines. The selected provider receives the prompt over HTTPS. Built-in workspace tools can list directories, read files, search with ripgrep, create/replace files (including new nested directories), inspect Git status/diffs, and propose a shell command. Commands always require an interactive approval; commands matching destructive/network patterns get an explicit ⚠ warning in the approval prompt. Every tool call is gated by a deterministic policy engine before it runs; `.env*`, `.git`, and `.febo` paths are protected at any depth, and symlinks may not escape the workspace. Repository hooks and MCP servers stay disabled unless explicitly enabled per run. Remote MCP, plugins, and subagents are not available yet.
