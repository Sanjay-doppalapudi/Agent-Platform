# harness

Minimal, fast coding agent CLI. OpenAI-compatible providers only (OpenCode Go, OpenRouter, NVIDIA NIM, any base_url). Six tools: `read` · `write` · `edit` · `bash` · `glob` · `grep`. Zero runtime dependencies — Bun built-ins only.

## Setup

```sh
bun install                       # dev types only
cp harness.config.example.json ~/.harness/config.json   # or a project-local harness.config.json
# set the API key env var your provider entry names, e.g.:
#   OPENCODE_GO_API_KEY / OPENROUTER_API_KEY / NVIDIA_API_KEY
```

Requires `rg` (ripgrep) on PATH for grep/glob.

## Usage

```sh
harness                          # interactive REPL (:new :resume <id> :model <id> :exit)
harness run -p "task" [--json]   # one-shot; --json emits NDJSON AgentEvents
harness serve [--port 4141]      # HTTP server mode
harness tool grep '{"pattern":"foo"}'   # run one tool directly (testing)

# ad-hoc endpoint, no config needed:
harness run -p "task" --base-url https://api.example.com/v1 --api-key sk-... -m some-model
```

Common flags: `--provider <name>` `-m/--model <id>` `--cwd <dir>` `--session <id>` `-c/--continue`.

Compile a single-file exe (~45ms cold start):

```sh
bun run compile   # → harness.exe
```

## Config

`harness.config.json` (walked up from cwd) or `~/.harness/config.json`. Precedence: CLI flags > `HARNESS_PROVIDER`/`HARNESS_MODEL`/`HARNESS_BASE_URL`/`HARNESS_API_KEY` env > project config > home config.

| Key | Default | Notes |
|---|---|---|
| `provider` / `providers` | — | each entry: `baseUrl`, `apiKeyEnv` (or `apiKey`), `model`, optional `cacheControl` (OpenRouter Anthropic-style breakpoints), `headers` |
| `permissions` | `"yolo"` | `"prompt"` asks y/N for write/edit/bash in the REPL |
| `maxIterations` | 40 | agent loop guard |
| `contextBudgetChars` | 400000 | old tool results elided beyond this |
| `redactEnv` | true | `.env` values read as `KEY=***` |
| `shell` | `"auto"` | Git Bash if found, else PowerShell (`bash`/`powershell`/`cmd` to force) |
| `parallelPolicy` | `"safe"` | read-only tools parallel, mutations serial (`all`/`none`) |
| `ignore` | `[]` | extends the hard ignore list (node_modules, .git, dist*, builds, trash, uploads, .vercel, .claude, temp, site-state) |

Drop a `HARNESS.md` in a project root — its content is appended to the system prompt for that project.

## Server API (for programmatic drivers)

- `GET /health` → `{ok, version, provider, model}`
- `POST /session {cwd?, title?}` → `{id}`
- `POST /session/:id/message {text, system?, model?, response_format?}` → blocks → `{text, messages}`
- `GET /event` · `GET /session/:id/events` → SSE of AgentEvents (`text`/`tool_start`/`tool_end`/`turn_end`/`done`/`error`, each with `sessionId`)
- `GET /session/:id/messages` → raw message array
- `DELETE /session/:id`

Sessions persist as append-only JSONL under `~/.harness/sessions/`.

## Why it's fast

- **Stable prompt prefix**: byte-identical system prompt + fixed-order terse tool schemas (~1K tokens total) → provider automatic prefix caching hits from turn 2 (verified: ~85% of prompt tokens cached).
- **Streaming always**, text rendered per-delta; tool_call deltas assembled by index.
- **Parallel reads**: multiple read/glob/grep calls in one turn run concurrently.
- **ripgrep everywhere**: grep *and* glob use `rg` so ignored dirs are pruned during traversal, not filtered after.
- **Hard output caps**: read 50KB, bash 30KB middle-out, grep 200 matches, glob 300 files, 40KB backstop per tool result.
- **No permission prompts** by default (`yolo`).
- **~45ms** compiled cold start; lazy per-mode imports.
