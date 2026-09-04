# Junebug plugin protocol

A plugin is any local executable that Junebug can drive as an alternative to
its own REST providers or the hardcoded `claude-cli`/`codex-cli` delegates
(`src/cli_delegate.rs`). Junebug's own code never needs to know what a
plugin actually does internally — in particular, it never contains any code
that touches the Claude Agent SDK, an OAuth token, or any other credential a
plugin might use. Junebug only ever: spawns the configured executable, writes
one JSON object to its stdin, and reads one JSON object back from its stdout.

This is intentionally the same shape `cli_delegate.rs` already uses for
`claude`/`codex` (run once, capture output, parse a final result) — a plugin
runs its **own** complete agent loop internally. Junebug's tool loop is not
involved: `tools` are never sent to a plugin, and a plugin's reply can never
contain tool calls Junebug executes itself.

## Manifest

`~/.junebug/plugins/<name>.json` (a workspace `.junebug/plugins/<name>.json`
overrides it for that workspace):

```json
{
  "command": "/absolute/path/to/executable",
  "args": ["optional", "extra", "argv"]
}
```

- `command` — required, an absolute path (or anything resolvable via `PATH`)
  to the executable to run.
- `args` — optional, extra fixed arguments passed before the protocol I/O
  described below (the prompt is never a CLI argument — see Request).

The manifest is re-read on every turn, so editing it takes effect on the
next message without restarting Junebug.

## Selecting a plugin

- `junebug --provider plugin --model <name>`
- `/model plugin:<name>` inside the REPL
- `/model plugin` opens a picker listing every configured manifest name
  (from both the workspace and user plugin directories)

`<name>` is the manifest's filename without `.json` — it is **not** a real
model name, unlike every other provider.

## Request (Junebug → plugin, written to stdin, then stdin is closed)

One JSON object:

```json
{
  "prompt": "the latest user message's text",
  "permission": "read-only | ask | workspace-write | yolo",
  "workspace": "/absolute/path/to/the/workspace/root"
}
```

- `prompt` — only the latest user message's text is sent, not the full
  conversation history — same design choice `cli_delegate.rs` makes for
  `claude`/`codex`: a plugin is expected to keep its own continuity
  (session/thread) on its own side if it wants multi-turn memory, rather
  than Junebug replaying its whole history as prompt text every turn.
- `permission` — Junebug's current permission mode, passed through as a
  plain string. A plugin that can write files or run commands should map
  this onto its own sandbox the same way `cli_delegate::effective_sandbox`
  does: `ask` collapses to the safe (read-only) side, since a plugin run
  this way is non-interactive and cannot be asked a real per-action
  approval question.
- `workspace` — the absolute path Junebug was started in. A plugin should
  treat this as its working directory (Junebug also sets the process's
  `cwd` to it, so a plugin that just inherits `cwd` needs nothing extra
  here).

## Response (plugin → Junebug, plugin's entire stdout, then it exits)

Stdout must be **exactly one JSON object** — nothing else on stdout before
or after it. Use stderr for logs/diagnostics; stderr is never parsed as the
response, only shown (truncated) if parsing stdout as JSON fails, to help
debug a broken plugin.

```json
{
  "text": "the final answer to show the user",
  "input_tokens": 0,
  "output_tokens": 0,
  "is_error": false
}
```

- `text` — required in spirit (defaults to `""` if omitted, which reads as
  an empty reply).
- `input_tokens` / `output_tokens` — optional (default `0`), shown in
  Junebug's `✓ <n>s · <model> (<provider>) · tokens N↑ M↓` turn summary.
  Best-effort; `0` is fine if a plugin has no way to know.
- `is_error` — optional (default `false`). When `true`, `text` is surfaced
  to the user as an error message instead of a normal assistant reply.

## Timeout and cancellation

- A plugin process is killed (whole process tree, `kill_command_tree`) if it
  runs longer than 10 minutes, or if the user interrupts the turn
  (Esc/Ctrl-C) — same behavior and timeout as `cli_delegate.rs`.
- On either, the plugin simply receives `SIGKILL` (or `taskkill /T /F` on
  Windows) with no warning on stdin; write a response only on a clean exit.

## Minimal reference implementation (Python)

```python
#!/usr/bin/env python3
import json
import sys

request = json.load(sys.stdin)
prompt = request["prompt"]

# ... call whatever you want here (e.g. the Claude Agent SDK) ...
answer_text = "you said: " + prompt

print(json.dumps({
    "text": answer_text,
    "input_tokens": 0,
    "output_tokens": 0,
    "is_error": False,
}))
```

Make it executable (`chmod +x`) and point a manifest's `command` at it.

## Why this exists

Anthropic's Agent SDK docs state that third-party developers cannot offer
`claude.ai` subscription login for their own products without prior
approval. Junebug is open source and distributed — so this repository
itself must never contain code that authenticates against a personal
subscription. The plugin protocol is the boundary that makes that possible
while still letting *you personally* wire your own subscription in: the
Agent SDK integration (if you build one) lives in a **separate,
non-distributed** program you write and run only for yourself, and that
program is the only thing that ever sees a token. Junebug just runs it and
reads JSON off its stdout, the same way it already runs `rg` or `ollama` or
`claude`/`codex` — a generic subprocess boundary, not a distribution
mechanism.
