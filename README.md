# Agent Platform (`ap`)

Minimal, fast coding agent CLI. OpenAI-compatible providers only (OpenCode Go, OpenRouter, NVIDIA NIM, any base_url). Six tools: `read` · `write` · `edit` · `bash` · `glob` · `grep`. Zero runtime dependencies — Bun built-ins only. ~45ms cold start.

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
| `permissions` | `"yolo"` | `"prompt"` asks y/N for write/edit/bash in the REPL |
| `maxIterations` | 40 | agent loop guard |
| `contextBudgetChars` | 400000 | old tool results elided beyond this |
| `redactEnv` | true | `.env` values read back as `KEY=***` |
| `shell` | `"auto"` | Git Bash if found, else PowerShell (`bash`/`powershell`/`cmd`) |
| `parallelPolicy` | `"safe"` | read-only tools run parallel, mutations serial (`all`/`none`) |
| `ignore` | `[]` | extends the hard ignore list (node_modules, .git, dist*, builds, trash, uploads, …) |

**API keys** live apart from config in `<dataDir>/credentials.json`, file-ACL'd to your user (`ap auth <provider>`). Resolution: `--api-key` → `HARNESS_API_KEY` → config `apiKey` → provider env var → credential store.

**Per-project prompt notes:** drop an `AP.md` (or legacy `HARNESS.md`) in a project root — its content is appended to the system prompt for that project (build rules, submodule conventions, dev-server ports…).

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

GitHub Actions cross-compiles all four platform binaries (`ap-windows-x64.exe`, `ap-linux-x64`, `ap-darwin-arm64`, `ap-darwin-x64`) and attaches them to the release; the install one-liners always fetch the latest release. To publish on the npm registry: the package name is `agentplatform` — `npm login && npm publish`, after which `npm i -g agentplatform` / `bun add -g agentplatform` / `yarn global add agentplatform` work everywhere bun is installed.
