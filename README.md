# Junebug CLI

Junebug CLI is an open-source, local-first coding agent for Junebug models.

This initial foundation intentionally defaults to read-only workspace access. It includes a provider-neutral event contract, safe read/search tools, and streaming REST support for OpenAI, OpenRouter, DeepSeek, Anthropic Claude, and local Ollama models.

Install on macOS (builds from source, installs to `~/.local/bin/junebug`):

```sh
./install-macos.sh
```

Tagged versions (`v*`) publish prebuilt binaries for macOS (arm64/x86_64), Linux, and Windows on the [GitHub releases page](https://github.com/weeksdev/junebug-cli/releases).

Save a provider API key once (stored in `~/.junebug/credentials.env`, `0600`) and start chatting. Existing `.febo` credentials, configuration, sessions, swarm setup, and checkpoints remain readable after the rename; all new state is written under `.junebug`. When you omit `--provider` and `--model`, Junebug defaults to the provider **and model** from your last session in this workspace (falling back to whichever key it can find and the provider's default model):

```sh
junebug set --provider deepseek YOUR_API_KEY
```

Or just run `junebug` with nothing configured: it starts in no-model mode and `/keys` sets a key from inside the REPL (arrow-key provider picker, hidden input) and brings a model up on the spot — no restart. Keys never appear in session logs and are never exposed to the model.

```sh
cargo run -- --help
OPENROUTER_API_KEY=... cargo run -- "Describe this project"
OPENAI_API_KEY=... cargo run -- --provider openai --model gpt-4.1-mini "Describe this project"
DEEPSEEK_API_KEY=... cargo run -- --provider deepseek --model deepseek-v4-flash "Describe this project"
ANTHROPIC_API_KEY=... cargo run -- --provider anthropic --model claude-sonnet-4-5 "Describe this project"
ollama pull qwen3:8b
cargo run -- --provider ollama --model qwen3:8b "Describe this project"
```

Ollama is detected automatically at `http://127.0.0.1:11434`; set `OLLAMA_HOST` for another local or LAN endpoint. It needs no Junebug credential. Installed Ollama models appear alongside cloud models in `/model` and `/swarm-setup`, and `ollama:model-name` works for direct selection. Junebug uses Ollama's OpenAI-compatible streaming chat, model-list, and tool-calling APIs and disables Qwen's long thinking trace for responsive agent turns. The default `qwen3:8b` is small enough for a 16 GB Apple Silicon Mac and supports real tool calls; plain code-completion models may only print tool-shaped text and are not suitable for agent mode.

Other local OpenAI-compatible runtimes are supported as `local-openai`. Set `LOCAL_OPENAI_BASE_URL` to the server root (for example an LM Studio, vLLM, or llama.cpp endpoint) and optionally set `LOCAL_OPENAI_API_KEY`; Junebug discovers `/v1/models` and uses `/v1/chat/completions`. Aliases `lmstudio`, `vllm`, and `openai-local` are accepted.

Use `/model` to pick from one grouped list containing the live model catalogs of every available provider; switching providers does not require a separate provider step. `provider:model` remains available for direct selection. See [PLAN.md](PLAN.md) for the release roadmap.

Automatic in-task routing is strictly opt-in: use `--model auto`, `/model auto`, or set `routing.mode` to `"auto"` in `~/.junebug/config.json` (a workspace `.junebug/config.json` overrides it). Configure `routing.routes` with band entries containing `provider` and `model`. Only available providers—credentialed cloud APIs or a reachable Ollama runtime—are offered. By default Junebug sends the routing service derived task signals—not prompts, code, filenames, diffs, or tool output. Set `routing.send_prompt` to `true` only if you explicitly want the prompt sent. The default service URL is `http://127.0.0.1:8791`; if it is unavailable, Junebug prints a notice and uses local routing rules.

The CLI uses standard provider environment variables and also reads an ignored `.env` file from the current workspace. Copy `.env.example` to `.env`; values are never printed or persisted in session logs.

## Current commands

- `junebug` — interactive REPL with a Claude Code-style terminal UI: streamed Markdown rendering, a spinner with elapsed time, `⏺ tool(args)` / `⎿ result` activity lines, and **Esc to interrupt** a running turn (the partial reply stays in context; your next message continues from it).
  - Slash commands: `/help`, `/keys` (set or replace a cloud-provider API key, input hidden), `/model` (arrow-key pick across every available provider's live model list, including Ollama), `/permissions` (arrow-key switch between read-only / ask / workspace-write / **yolo** mid-session), `/rewind` (restore workspace files to an earlier checkpoint), `/compact` (model-written summary replaces old history), `/status`, `/changes` (full-screen changed-file tree and per-file diff, using real Git or the latest shadow checkpoint in a non-Git folder), `/explorer` (searchable read-only workspace tree with syntax-highlighted file viewer and explicit tree/file pane focus), `/diff`, `/exit`. A dimmed status line under the prompt always shows the current model and access level.
  - Input intellisense: typing `/` opens a slash-command menu; typing `@` opens a workspace file search menu (↑/↓ select, Tab/Enter accept). Accepted `@path` mentions attach the file's contents to your message.
  - Line editing: arrows, Home/End, Ctrl-A/E/U/K, and ↑/↓ history. Shift+Tab cycles `read-only → ask → workspace-write → yolo` without losing typed input and also works during a running turn; the colored footer/spinner always shows the effective mode. Each in-flight tool snapshots the mode when it begins, so a change affects the next tool call rather than altering one halfway through.
- `junebug [prompt]` — single prompt execution with approval prompts for writes and commands.
- `junebug exec [--json] <prompt>` — script-friendly execution emitting JSON Lines events (`text.delta`, `tool.call`, `tool.result`, `route.selected`, `route.changed`, `completed`).
- `--permission read-only|ask|workspace-write|yolo` — defaults to `read-only`. `yolo` grants unrestricted filesystem access—including absolute/outside-workspace paths and normally protected `.env*`, `.git`, `.junebug`, and legacy `.febo` paths—lets commands inherit the launching environment, and approves every write and command without asking (plan mode still forces read-only). This can send protected files or environment secrets to the provider and local session log. Search, Git status, and Git diff accept an optional target path for work outside the startup repository. Checkpoints only cover the startup workspace, so outside-workspace changes are not rewindable. Non-interactive runs require `workspace-write` or `yolo` for edits.
- `--plan` — hard read-only guard that also hides write/command tools from the model, regardless of `--permission`.
- `--resume [session.jsonl]` — continue a session; **with no path it lists past sessions to pick from**. `--resume-compact [session.jsonl]` additionally summarizes a large history first so resuming does not waste tokens. `--max-context-chars` bounds the request context with deterministic compaction.
- **Model swarms** — `/swarm-setup` assigns models from one grouped all-provider picker to three roles: a **boss** that inspects the workspace, writes a constitution + task plan, rules on disputes, and reviews the result (use your strongest model — it never writes files); a **worker** that executes each task (cheap); and a **checker** that independently verifies every task with its own tools, never trusting the worker's report (cheap). `/swarm <goal>` runs the loop: failed checks send specific feedback back to the worker (up to 2 reworks), persistent disputes escalate to the boss — which can overrule the checker — and the boss closes with an honest final report. Every step is labeled with the role and model doing it, each swarm gets its own session log, and checkpoints are taken throughout so `/rewind` can undo a swarm. Requires `ask` or `workspace-write` permission (`yolo` avoids per-step prompts).
- **File-change diffs** — every file write shows what changed: colored `-`/`+` line hunks with context appear in the REPL activity stream under the tool line, in `ask`-mode approval prompts (so you see exactly what you're approving before answering y/N), and as `file.diff` events in `exec --json`. Diffs are display-only and are never sent to the model.
- **Workspace checkpoints** — before every prompt, file write, and command, Junebug snapshots the workspace into a shadow Git repository under `~/.junebug/checkpoints/` (existing `~/.febo/checkpoints/` histories remain usable; your own repo is never touched; works in non-Git workspaces). `/rewind` restores files to any checkpoint, and the pre-restore state is checkpointed first so a rewind is always undoable. Snapshots respect your `.gitignore` and never capture `.env*`, `.junebug/`, legacy `.febo/`, or build caches. Disable with `--no-checkpoints`.
- `--enable-hooks` — run explicitly trusted `.junebug/hooks.json` lifecycle commands.
- `--enable-mcp` — connect local stdio MCP servers from `.junebug/mcp.json`; every MCP tool call requires an interactive approval showing the arguments.
- `junebug --version` / `junebug --help`.

Sessions are recorded locally in `.junebug/sessions/` as JSON Lines, and legacy `.febo/sessions/` remain resumable. Cloud providers receive prompts over HTTPS; Ollama requests stay on the configured `OLLAMA_HOST` endpoint. Built-in workspace tools can list directories, read files, search with ripgrep, create/replace files (including new nested directories), inspect Git status/diffs, and propose a shell command. Git inspection reports cleanly when the workspace is not a Git repository. Outside `yolo`, commands require interactive approval and destructive/network patterns get an explicit ⚠ warning in the approval prompt; yolo bypasses both. Every tool call is gated by a deterministic policy engine before it runs. Outside `yolo`, `.env*`, `.git`, `.junebug`, and legacy `.febo` paths are protected at any depth and symlinks may not escape the workspace. Repository hooks and MCP servers stay disabled unless explicitly enabled per run. Remote MCP, plugins, and subagents are not available yet.
