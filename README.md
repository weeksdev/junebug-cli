# Febo CLI

Febo CLI is an open-source, local-first coding agent for Febo models.

This initial foundation intentionally defaults to read-only workspace access. It includes a provider-neutral event contract, safe read/search tools, and streaming REST support for OpenAI, OpenRouter, and DeepSeek.

Install on macOS (builds from source, installs to `~/.local/bin/febo`):

```sh
./install-macos.sh
```

Tagged versions (`v*`) publish prebuilt binaries for macOS (arm64/x86_64), Linux, and Windows on the [GitHub releases page](https://github.com/weeksdev/febo_cli/releases).

Save a provider API key once (stored in `~/.febo/credentials.env`, `0600`) and start chatting:

```sh
febo set --provider deepseek YOUR_API_KEY
```

```sh
cargo run -- --help
OPENROUTER_API_KEY=... cargo run -- "Describe this project"
OPENAI_API_KEY=... cargo run -- --provider openai --model gpt-4.1-mini "Describe this project"
DEEPSEEK_API_KEY=... cargo run -- --provider deepseek --model deepseek-v4-flash "Describe this project"
```

OpenRouter is the default provider. Use `--model` to select a model supported by the provider. See [PLAN.md](PLAN.md) for the release roadmap.

The CLI uses standard provider environment variables and also reads an ignored `.env` file from the current workspace. Copy `.env.example` to `.env`; values are never printed or persisted in session logs.

## Current commands

- `febo` — interactive REPL with a Claude Code-style terminal UI: streamed Markdown rendering, a spinner with elapsed time, `⏺ tool(args)` / `⎿ result` activity lines, and **Esc to interrupt** a running turn (the partial reply stays in context; your next message continues from it).
  - Slash commands: `/help`, `/model [NAME]` (lists available models from the provider's `/models` endpoint; switches with fuzzy matching), `/compact` (model-written summary replaces old history), `/status`, `/diff`, `/exit`.
  - Input intellisense: typing `/` opens a slash-command menu; typing `@` opens a workspace file search menu (↑/↓ select, Tab/Enter accept). Accepted `@path` mentions attach the file's contents to your message.
  - Line editing: arrows, Home/End, Ctrl-A/E/U/K, and ↑/↓ history.
- `febo [prompt]` — single prompt execution with approval prompts for writes and commands.
- `febo exec [--json] <prompt>` — script-friendly execution emitting JSON Lines events (`text.delta`, `tool.call`, `tool.result`, `completed`).
- `--permission read-only|ask|workspace-write` — defaults to `read-only`; non-interactive runs require `workspace-write` for edits because approval prompts are unavailable.
- `--plan` — hard read-only guard that also hides write/command tools from the model, regardless of `--permission`.
- `--resume <session.jsonl>` — continue a recorded session; `--resume-compact <session.jsonl>` additionally summarizes a large history first so resuming does not waste tokens. `--max-context-chars` bounds the request context with deterministic compaction.
- `--enable-hooks` — run explicitly trusted `.febo/hooks.json` lifecycle commands.
- `--enable-mcp` — connect local stdio MCP servers from `.febo/mcp.json`; every MCP tool call requires an interactive approval showing the arguments.
- `febo --version` / `febo --help`.

Sessions are recorded locally in `.febo/sessions/` as JSON Lines. The selected provider receives the prompt over HTTPS. Built-in workspace tools can list directories, read files, search with ripgrep, create/replace files (including new nested directories), inspect Git status/diffs, and propose a shell command. Commands always require an interactive approval; commands matching destructive/network patterns get an explicit ⚠ warning in the approval prompt. Every tool call is gated by a deterministic policy engine before it runs; `.env*`, `.git`, and `.febo` paths are protected at any depth, and symlinks may not escape the workspace. Repository hooks and MCP servers stay disabled unless explicitly enabled per run. Remote MCP, plugins, and subagents are not available yet.
