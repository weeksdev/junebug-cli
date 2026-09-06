# Junebug custom tool protocol

A custom tool is a single named capability backed by an external executable
— the small-grained sibling of [`PLUGIN_PROTOCOL.md`](PLUGIN_PROTOCOL.md)
(which delegates a whole conversational turn to an external agent). A tool
call carries structured arguments and returns a text result, the same
shape every builtin tool (`read_file`, `run_command`, ...) already has —
custom tools just let you add your own.

Custom tools are meant to be **produced and tested conversationally** via
`/tool-build`, not hand-written: describe what you need, the builder writes
a script, actually test-runs it against the protocol below before
proposing it as done, and only then is it registered. You can still write
one by hand if you prefer; nothing about the protocol requires the
conversational path.

## Manifest

`~/.junebug/tools/<name>.json` (global) or `.junebug/tools/<name>.json`
(repo, overrides a global tool of the same name):

```json
{
  "name": "count_lines",
  "description": "Counts the number of lines in a file...",
  "parameters": {
    "type": "object",
    "properties": { "path": { "type": "string", "description": "..." } },
    "required": ["path"]
  },
  "command": "/absolute/path/to/count_lines.py",
  "args": []
}
```

- `name` — the callable tool name a model will see and invoke. Should look
  like a function name (`count_lines`, not "Count Lines" or a sentence).
- `description` — written for a model deciding whether and how to call the
  tool, same as a builtin tool's description.
- `parameters` — a JSON Schema `object`, the exact same shape a builtin
  tool's `function.parameters` already uses.
- `command` / `args` — the executable and any fixed extra arguments.

A custom tool is always classified `ToolRisk::Execute` — the same
treatment MCP tools already get, since its actual risk is unknowable from
its name — so outside `yolo` it always requires an explicit approval, and
plan mode never sees it at all (plan mode's guarantee of zero side effects
is incompatible with an arbitrary script of unknown risk).

## Request (Junebug → tool, written to stdin, then stdin is closed)

One JSON object:

```json
{
  "arguments": { "path": "app.js" },
  "workspace": "/absolute/path/to/the/workspace/root"
}
```

- `arguments` — exactly what the model passed for the call, matching your
  `parameters` schema.
- `workspace` — the workspace root; the process's `cwd` is also set to
  this, so a script that just uses relative paths from its own `cwd` needs
  nothing extra here.

## Response (tool → Junebug, exactly the tool's stdout, then it exits)

Stdout must be **exactly one JSON object** — nothing else on stdout before
or after it. Use stderr for logs/diagnostics; stderr is only ever shown
(truncated) if parsing stdout as JSON fails, to help debug a broken tool.

```json
{ "result": "3 lines", "is_error": false }
```

- `result` — the text result shown to the model, exactly like any builtin
  tool's return value.
- `is_error` — when `true`, `result` is surfaced as an `ERROR:`-prefixed
  result instead of a normal one (matching every builtin tool's own error
  convention).

## Timeout and cancellation

Killed (whole process tree) if it runs longer than the same ceiling
`run_command` uses (`MAX_COMMAND_TIMEOUT_SECS`, currently one hour) — a
tool call has no separate timeout knob of its own the way `run_command`
does.

## Minimal reference implementation (Python)

```python
#!/usr/bin/env python3
import json, sys

request = json.load(sys.stdin)
path = request["arguments"]["path"]
workspace = request["workspace"]

try:
    with open(path if path.startswith("/") else f"{workspace}/{path}") as f:
        count = sum(1 for _ in f)
    result, is_error = f"{count} lines", False
except OSError as error:
    result, is_error = str(error), True

print(json.dumps({"result": result, "is_error": is_error}))
```

Make it executable (`chmod +x`) and point a manifest's `command` at it —
or just have `command` be an interpreter (`python3`) and put the script
path in `args`, as `/tool-build` typically does.

## Why "produced and tested," not just "produced"

A model that writes a script and never runs it is exactly as likely to get
the protocol subtly wrong (a stray print statement polluting stdout, an
off-by-one, wrong JSON key names) as any other unverified code. `/tool-build`
requires actually invoking the script it just wrote (via `run_command`,
with a realistic sample request piped to stdin) and confirming the output
matches this contract *before* proposing the tool as done — the same
"don't trust an unverified claim" principle `/swarm`'s checker role and
`/investigate`'s skeptic role already apply to code and reasoning
respectively, applied here to a tool's own protocol compliance.
