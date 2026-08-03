<p align="center"><img src="logo.svg" width="200" alt="Agent Platform"></p>

# Agent Platform (`ap`)

Minimal, fast coding agent CLI. OpenAI-compatible providers only (OpenCode Go, OpenRouter, NVIDIA NIM, any base_url). Core tools: `read` · `write` · `edit` · `bash` · `glob` · `grep` (+ `agent` · `fetch` · `todo` · `websearch` in the full profile). Zero runtime dependencies — Bun built-ins only. ~45ms cold start.

```
◆ AP · opencode-go/minimax-m3
  cwd C:\projects\my-app
  session 20260801...-ab3d · type / for commands · ctrl+o details · ctrl+c abort

code › fix the failing date test
✻ reasoning streams dim…
✓ grep "formatDate" (*.ts) · 4 hits · 22ms
╭ src/utils/date.ts
- return d.toLocaleDateString()
+ return d.toISOString().slice(0, 10)
╰
✓ edit src/utils/date.ts · replaced 1 · 1ms
✓ bash npm test · ok · 2.1s
Fixed — the formatter now returns ISO dates; tests pass.

3 steps · ↑4.1k (3.7k cached) ↓240 · 6.2s
```

## Install

**One-liner binary install (no runtime needed — pulls the latest GitHub release):**

```powershell
# Windows
irm https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.ps1 | iex
```
```sh
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.sh | bash
```

**Via a package manager, straight from GitHub (needs [bun](https://bun.sh) on the machine):**

```sh
bun add -g github:Sanjay-doppalapudi/Agent-Platform
npm  i  -g github:Sanjay-doppalapudi/Agent-Platform
yarn global add Sanjay-doppalapudi/Agent-Platform
pnpm add -g github:Sanjay-doppalapudi/Agent-Platform
bunx github:Sanjay-doppalapudi/Agent-Platform --help   # one-off, no install
```

**From source:**

```sh
git clone https://github.com/Sanjay-doppalapudi/Agent-Platform && cd Agent-Platform
bun install && bun link
```

Every install also needs [ripgrep](https://github.com/BurntSushi/ripgrep) on PATH (`winget install BurntSushi.ripgrep.MSVC` / `brew install ripgrep` / `apt install ripgrep`).

**First run:**

```sh
ap auth opencode-go        # store your API key (hidden input, user-locked file)
cd your-project && ap      # go
```

## Usage

```sh
ap                      # interactive REPL in the current directory
ap run -p "task"        # one-shot; --json emits NDJSON AgentEvents
ap serve [--port 4141]  # HTTP server mode (sessions + SSE)
ap models [query]       # search the models.dev catalog (context + pricing)
ap auth <provider>      # store an API key
ap prompt [--cwd dir]   # print the system prompt used for a directory
ap tool grep '{"pattern":"foo"}'   # run one tool directly (testing)
```

Common flags: `--provider <name>` · `-m/--model <id>` · `--cwd <dir>` · `--mode plan|code` (`--plan`) · `--session <id>` · `-c/--continue` · `--base-url <url> --api-key <key>` for any ad-hoc endpoint.

### REPL

Type `/` to open the command menu (↑/↓ navigate, Enter/Tab select, Esc close; ↑/↓ recall history otherwise):

| Command | Effect |
|---|---|
| `/plan` / `/code` | switch mode — plan is structurally read-only (only read/glob/grep schemas are sent) and produces an implementation plan; code (default) has all tools. The prompt shows the active mode: `plan › ` / `code › ` |
| `/model <id>` | switch model; `/model <provider>/<model>` switches provider too — unknown providers are resolved live from models.dev |
| `/models <query>` | search the models.dev catalog |
| `/new` `/resume <id>` `/sessions` | session management |
| `/system` `/context` | inspect the system prompt / token usage |
| `/exit` | quit (prints the session id + resume command) |

**Ctrl+O** toggles details (reasoning + diffs). While off, details are buffered (60KB cap) and replayed when you toggle back on. **Ctrl+C** aborts the running turn; at an empty prompt it exits.

Every edit/write renders a diff before it runs, built from the tool args (no extra I/O):

```
╭ hi.html
- <h1>Hello</h1>
+ <h1>Goodbye</h1>
╰
```

## Configuration

Project `ap.config.json` (walked up from cwd; legacy `harness.config.json` accepted) or `<dataDir>/config.json`. Data dir: `~/.ap` (existing `~/.harness` dirs keep working). Precedence: CLI flags > `HARNESS_PROVIDER`/`HARNESS_MODEL`/`HARNESS_BASE_URL`/`HARNESS_API_KEY` env > project config > home config.

```jsonc
{
  "provider": "opencode-go",
  "providers": {
    "opencode-go": { "baseUrl": "https://opencode.ai/zen/go/v1", "apiKeyEnv": "OPENCODE_GO_API_KEY", "model": "minimax-m3" },
    "openrouter":  { "baseUrl": "https://openrouter.ai/api/v1", "apiKeyEnv": "OPENROUTER_API_KEY", "model": "...", "cacheControl": true }
  }
}
```

| Key | Default | Notes |
|---|---|---|
| `mode` | `"code"` | `"plan"` = read-only tools |
| `permissions` | `"yolo"` | `"prompt"` asks before every mutating tool in the REPL |
| `sandbox` | `"workspace"` | writes/edits outside the workspace (+ data dir + session plans) need a y/N/always permission; `"off"` or `--no-sandbox` disables; headless denies unless `--allow-outside` |
| `bashGuard` | `"on"` | dangerous shell patterns (recursive absolute deletes, format, registry edits, curl\|bash, …) are auto-blocked, warned, and logged to `<dataDir>/blocked-commands.jsonl` for provider feedback |
| `streamIdleSeconds` | 90 | stalled provider streams abort and retry once instead of hanging (0 = off) |
| `maxIterations` | 40 | agent loop guard |
| `contextBudgetChars` | 400000 | old tool results elided beyond this |
| `redactEnv` | true | `.env` values read back as `KEY=***` |
| `shell` | `"auto"` | Git Bash if found, else PowerShell (`bash`/`powershell`/`cmd`) |
| `parallelPolicy` | `"safe"` | read-only tools run parallel, mutations serial (`all`/`none`) |
| `ignore` | `[]` | extends the hard ignore list (node_modules, .git, dist*, builds, trash, uploads, …) |

**API keys** live apart from config in `<dataDir>/credentials.json`, file-ACL'd to your user (`ap auth <provider>`). Resolution: `--api-key` → `HARNESS_API_KEY` → config `apiKey` → provider env var → credential store.

**Per-project prompt notes:** drop an `AP.md` (or legacy `HARNESS.md`) in a project root — its content is appended to the system prompt for that project (build rules, submodule conventions, dev-server ports…).

## Sandbox

Reads are unrestricted; **mutations are jailed**: `write`/`edit` outside the workspace, the AP data dir, or the session plans folder trigger an interactive `[y/N/a=always]` permission in the REPL and are denied headlessly unless `--allow-outside`. Dangerous bash patterns are **auto-blocked** (never prompted) with a ⚠ warning and a JSONL log entry you can share with your model provider. This is a guardrail, not a VM: pattern scanning is best-effort, symlinks aren't resolved, and network egress isn't restricted. `/sandbox` shows state; `/sandbox off` disables per-session.

Weaker-model tolerance: tool names (`search`→grep, `create`→write, …) and argument names (`file_path`, `command`, `query`, …) are alias-normalized, malformed JSON args get auto-repaired, and edits retry with CRLF and trailing-whitespace normalization — each recovery saves a full model round-trip.

## Full-profile features (absent in `--light`)

- **Checkpoints**: every mutating turn auto-commits the workspace to a shadow git repo (your real git history is untouched; works in non-git folders). `/undo`, `/diff [n]`, `/checkpoints`, `/restore <hash>`.
- **Subagents**: the `agent` tool delegates independent subtasks to parallel child processes (`ap run --light` under the hood — children can't recurse). Live nested `↳` progress lines stream in, and `/agents` lists every subagent with status, steps, and duration.
- **Custom slash commands**: drop `.ap/commands/<name>.md` in a repo (or `<dataDir>/commands/`) — `/name args` expands the file as your message with `$ARGS` substitution; appears in the `/` menu automatically.
- **`@file` mentions**: `@src/foo.ts` in a message inlines the file (8KB cap) — saves the model a read turn.
- **Hooks**: config `hooks.preBash/preWrite/preEdit` (nonzero exit blocks the tool with the hook's output) and `hooks.afterEdit` (e.g. `bun x tsc --noEmit` — failures are fed back so the model fixes them itself, max 2 rounds/turn).
- **`websearch` tool**: web search via DuckDuckGo's HTML endpoint (plain fetch, no API key) — titles, URLs, snippets. **`fetch` tool**: URL → readable text (50KB cap); `render:true` runs the page through your *installed* Chrome/Edge headless (`--dump-dom`, ~300ms, nothing bundled) so JS-rendered pages work — falls back to plain fetch if no browser is found. **`todo` tool**: session checklist rendered live in the transcript.
- **`/worktree new <slug> | list | back | merge <slug>`**: isolated git worktree + `ap/<slug>` branch per task — parallel work never collides.
- **`/compact`**: summarizes the session into a fresh one when context grows.
- **`AGENTS.md`** project notes supported alongside `AP.md`/`HARNESS.md`; `ap resume` (interactive picker) and `ap sessions search <q>` (ripgrep over transcripts).

## Sessions (no database)

Each session is one append-only JSONL file in `<dataDir>/sessions/<id>.jsonl` — a `meta` line, then one line per message, flushed on every append. Crash-safe (torn last line ignored), resumable (`-c`, `--resume <id>`, `/resume`), greppable, and deleted by deleting the file.

## Server API (for programmatic drivers)

- `GET /health` → `{ok, version, provider, model}`
- `POST /session {cwd?, title?}` → `{id}`
- `POST /session/:id/message {text, system?, model?, response_format?}` → blocks → `{text, messages}`
- `GET /event` · `GET /session/:id/events` → SSE of AgentEvents (`text`/`reasoning`/`tool_start`/`tool_end`/`turn_end`/`done`/`error`, each with `sessionId`)
- `GET /session/:id/messages` → raw message array
- `DELETE /session/:id`

## Why it's fast

- **Stable prompt prefix**: byte-identical system prompt + fixed-order terse tool schemas (~1K tokens total) → provider automatic prefix caching hits from turn 2 (measured ~85–90% of prompt tokens cached).
- **Streaming always**; tool_call deltas assembled by index; reasoning (`<think>`/`reasoning_content`) separated from answers and never re-sent as history.
- **Parallel reads**: multiple read/glob/grep calls in one turn run concurrently.
- **ripgrep everywhere**: grep *and* glob use `rg`, so ignored dirs are pruned during traversal, not filtered after.
- **Hard output caps**: read 50KB · bash 30KB middle-out · grep 200 matches · glob 300 files · 40KB backstop per result.
- **No permission stalls** by default (`yolo`), no database, lazy per-mode imports, `bun build --compile` single binary.

## Releasing

```sh
git tag v0.1.0 && git push --tags
```

GitHub Actions cross-compiles all four platform binaries (`ap-windows-x64.exe`, `ap-linux-x64`, `ap-darwin-arm64`, `ap-darwin-x64`) and attaches them to the release; the install one-liners always fetch the latest release.

**Registry installs** (after `npm publish` — package is `@sanjaydoppalapudi/agentplatform`, command is still `ap`):

```sh
npm i -g @sanjaydoppalapudi/agentplatform
bun add -g @sanjaydoppalapudi/agentplatform
yarn global add @sanjaydoppalapudi/agentplatform
pnpm add -g @sanjaydoppalapudi/agentplatform
```
