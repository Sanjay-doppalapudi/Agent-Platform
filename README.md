<p align="center"><img src="logo.svg" width="200" alt="Agent Platform"></p>

# Agent Platform (`ap`)

Minimal, fast coding agent CLI. OpenAI-compatible providers only (OpenCode Go, OpenRouter, NVIDIA NIM, any base_url). Core tools: `read` · `write` · `edit` · `bash` · `glob` · `grep` (+ `agent` · `fetch` · `todo` · `websearch` in the full profile). Plugs into the existing ecosystem: **MCP client** (any MCP server's tools become agent tools), **ACP agent** (runs inside Zed), **skills.sh / Claude Code skills**, lifecycle **hooks/webhooks**. Zero runtime dependencies — Bun built-ins only. ~45ms cold start.

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
ap loop -p "goal"       # loop work→verify until the goal is verifiably done
ap skills               # list / install SKILL.md packs (skills.sh compatible)
ap mcp                  # connect MCP servers — their tools become agent tools
ap acp                  # ACP agent for editors (Zed): stdio, modes, permissions
ap serve [--port 4141]  # HTTP server mode (sessions + SSE)
ap models [query]       # search the models.dev catalog (context + pricing)
ap auth <provider>      # store an API key
ap resume / ap sessions # pick a session to resume / list + full-text search
ap prompt [--cwd dir]   # print the system prompt used for a directory
ap tool grep '{"pattern":"foo"}'   # run one tool directly (testing)
ap help <command>       # detailed help (run · loop · skills · mcp · acp · serve · sessions)
```

Common flags: `--provider <name>` · `-m/--model <id>` · `--cwd <dir>` · `--mode plan|code` (`--plan`) · `--session <id>` · `-c/--continue` · `--base-url <url> --api-key <key>` for any ad-hoc endpoint.

### REPL

Type `/` to open the command menu (↑/↓ navigate, Enter/Tab select, Esc close; ↑/↓ recall history otherwise):

| Command | Effect |
|---|---|
| `/plan` / `/code` | switch mode — plan is structurally read-only (only read/glob/grep schemas are sent) and produces an implementation plan; code (default) has all tools. The prompt shows the active mode: `plan › ` / `code › ` |
| `/model <id>` | switch model; `/model <provider>/<model>` switches provider too — unknown providers are resolved live from models.dev |
| `/theme [name]` | list themes with color swatches, or switch — `default` · `mono` · `nord` · `dracula` · `gruvbox` · `solarized` · `matrix`. The choice is saved to `<dataDir>/config.json`; `NO_COLOR` forces mono |
| `/effort low\|medium\|high\|off` | reasoning effort, sent as `reasoning_effort` (config default: `reasoningEffort`); checks models.dev whether the model supports reasoning and warns if not — shown on the status line |
| `/models <query>` | search the models.dev catalog |
| `/new` `/resume <id>` `/sessions` | session management |
| `/undo` `/diff [n]` `/checkpoints` `/restore <hash>` | shadow-git checkpoint ops (see below) |
| `/worktree` `/compact` `/agents` | worktree per task · summarize into fresh session · list subagents |
| `/mcp` `/skills` `/sandbox` | MCP server status · installed skills · sandbox state/toggle |
| `/system` `/context` | inspect the system prompt / token usage |
| `/exit` | quit (prints the session id + resume command) |

The prompt sits in a frame whose top edge labels the mode and whose bottom edge *is* the status line, so state is always visible without stealing a row:

```
╭─ code ───────────────────────────────────────────────────╮
│ › fix the failing date test
╰─ opencode-go/minimax-m3 · ctx 62% · ~$0.004 ─────────────╯
```

Output has a clean visual hierarchy: **answer text is flush-left** in the default color; everything that isn't the answer — tool lines (`✓` + cyan action + dim timing), diffs, dim `✻` reasoning, warnings — is indented two spaces. **Ctrl+O** toggles details (reasoning + subagent progress); diffs always show — a file never changes without the diff having been visible. While off, details are buffered and replayed when you toggle back on. **Ctrl+C** aborts the running turn; at an empty prompt it exits. The spinner shows live elapsed time and output tokens (`⠹ thinking · 12s · ~340 tok`), the per-turn stats line includes context usage (`ctx 42%`, with a `/compact` reminder at 60%), and errors come with a suggested fix (`fix: ap auth <provider>`). First-time moments teach their command: the first sandbox prompt points at `/sandbox`, the first subagent at `/agents`.

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
| `permission` | — | per-tool rules, evaluated first: `{"fetch": "deny", "edit": "ask", "mcp_*": "ask", "bash": {"git push*": "ask", "*": "allow"}}` — tool keys and bash command patterns take `*` globs; `deny` blocks outright, `ask` uses the interactive permit (auto-denied headless), `allow` skips only the ask gate (sandbox + bashGuard still apply) |
| `sandbox` | `"workspace"` | writes/edits outside the workspace (+ data dir + session plans) need a y/N/always permission; `"off"` or `--no-sandbox` disables; headless denies unless `--allow-outside` |
| `bashGuard` | `"on"` | dangerous shell patterns (recursive absolute deletes, format, registry edits, curl\|bash, …) are auto-blocked, warned, and logged to `<dataDir>/blocked-commands.jsonl` for provider feedback |
| `streamIdleSeconds` | 90 | stalled provider streams abort and retry once instead of hanging (0 = off) |
| `maxIterations` | 40 | agent loop guard |
| `contextBudgetChars` | 400000 | old tool results elided beyond this |
| `redactEnv` | true | `.env` values read back as `KEY=***` |
| `shell` | `"auto"` | Git Bash if found, else PowerShell (`bash`/`powershell`/`cmd`) |
| `parallelPolicy` | `"safe"` | read-only tools run parallel, mutations serial (`all`/`none`) |
| `ignore` | `[]` | extends the hard ignore list (node_modules, .git, dist*, builds, trash, uploads, …) |
| `checkpoints` | `"on"` | shadow-git checkpoint after every mutating turn (`"off"` disables) |
| `autoCompact` | `"on"` | REPL auto-summarizes into a fresh session at 85% of the context budget (`"off"` disables; `/compact` stays manual) |
| `theme` | `"default"` | color theme: `default`/`mono`/`nord`/`dracula`/`gruvbox`/`solarized`/`matrix` (also `/theme`) |
| `reasoningEffort` | — | `"low"`/`"medium"`/`"high"` — default `reasoning_effort` sent with every request (`/effort` and `--effort` override) |
| `hooks` | — | `preBash`/`preWrite`/`preEdit`/`afterEdit` tool hooks + `onDone`/`onError` lifecycle hooks (command or webhook URL) |
| `mcpServers` | — | MCP servers, Claude Code format (also read from a project `.mcp.json`) |

**API keys** live apart from config in `<dataDir>/credentials.json`, file-ACL'd to your user (`ap auth <provider>`). Resolution: `--api-key` → `HARNESS_API_KEY` → config `apiKey` → provider env var → credential store.

**Per-project prompt notes:** drop an `AP.md` (or legacy `HARNESS.md`) in a project root — its content is appended to the system prompt for that project (build rules, submodule conventions, dev-server ports…).

## Sandbox

**Reads are scoped**: the agent can freely read the workspace plus the skills/memory/commands dirs and its session plans folder. Reading anywhere else — other projects, your home dir, `ls`/`cat` via bash included (best-effort path scan, `..` escapes counted) — triggers the interactive `[y/N/a=always]` permission and is denied headlessly unless `--allow-outside`. **AP-private data is never accessible**: session transcripts, checkpoints, `credentials.json`, and `config.json` are hard-denied even if you'd approve — only `sandbox:"off"` lifts that. **Mutations are jailed** the same way: `write`/`edit` outside the workspace/data dir/plans folder need permission. Dangerous bash patterns are **auto-blocked** (never prompted) with a ⚠ warning and a JSONL log entry you can share with your model provider. This is a guardrail, not a VM: pattern scanning is best-effort, symlinks aren't resolved, and network egress isn't restricted. `/sandbox` shows state (writable + readable roots); `/sandbox off` disables per-session.

Weaker-model tolerance: tool names (`search`→grep, `create`→write, …) and argument names (`file_path`, `command`, `query`, …) are alias-normalized, malformed JSON args get auto-repaired, and edits retry with CRLF and trailing-whitespace normalization — each recovery saves a full model round-trip.

## Full-profile features (absent in `--light`)

- **Checkpoints**: every mutating turn auto-commits the workspace to a shadow git repo (your real git history is untouched; works in non-git folders). `/undo`, `/diff [n]`, `/checkpoints`, `/restore <hash>`.
- **Subagents**: the `agent` tool delegates independent subtasks to parallel child processes (`ap run --light` under the hood — children can't recurse). Live nested `↳` progress lines stream in, and `/agents` lists every subagent with status, steps, and duration.
- **Named agents** (opencode-style): drop `.ap/agents/<name>.md` (frontmatter `description` / `model` / `tools`; body = role instructions) — `.claude/agents/` works too. The model delegates with `{"name": "reviewer", "task": …}` (profiles are listed in the system prompt, one line each), `ap run --agent <name> -p …` runs one headlessly, and a profile's `tools:` list becomes a hard schema whitelist for that agent (e.g. a read-only reviewer literally cannot write).
- **Custom slash commands**: drop `.ap/commands/<name>.md` in a repo (or `<dataDir>/commands/`) — `/name args` expands the file as your message with `$ARGS` substitution; appears in the `/` menu automatically.
- **`@file` mentions**: `@src/foo.ts` in a message inlines the file (8KB cap) — saves the model a read turn.
- **Hooks**: config `hooks.preBash/preWrite/preEdit` (nonzero exit blocks the tool with the hook's output) and `hooks.afterEdit` (e.g. `bun x tsc --noEmit` — failures are fed back so the model fixes them itself, max 2 rounds/turn). **Lifecycle hooks**: `hooks.onDone` / `hooks.onError` fire when a turn finishes — either a shell command (JSON payload in `AP_PAYLOAD`, event name in `AP_EVENT`) or an `http(s)://` URL that receives a JSON POST (`{event, sessionId, cwd, text|message}`). Fire-and-forget: they never block or fail the turn (run/loop drain them before exiting so they always deliver). Notify Slack, kick a build, chain another `ap run` — anything that should happen "when the chat finishes".
- **`websearch` tool**: web search via DuckDuckGo's HTML endpoint (plain fetch, no API key) — titles, URLs, snippets. **`fetch` tool**: URL → readable text (50KB cap); `render:true` runs the page through your *installed* Chrome/Edge headless (`--dump-dom`, ~300ms, nothing bundled) so JS-rendered pages work — falls back to plain fetch if no browser is found. **`todo` tool**: session checklist rendered live in the transcript.
- **`/commit [message]`**: stages everything and commits to your **real** git history — the model drafts an imperative subject + why-body from the diff, you see it and confirm `y/N`, and nothing is ever pushed. On a protected branch (`main`, `master`, `develop`, `release/*`, …) it refuses to commit directly and offers to create an `ap/<slug>` branch first. Pass your own message to skip drafting.
- **`/worktree new <slug> | list | back | merge <slug>`**: isolated git worktree + `ap/<slug>` branch per task — parallel work never collides.
- **`/compact`**: summarizes the session into a fresh one — and runs **automatically at 85% of the context budget** (`autoCompact: "off"` disables), so long sessions never degrade into silently-elided history.
- **`/ps`** (also `ap ps`): background processes started with `bash background:true` (dev servers, watchers, long builds) are registered, so you can list them with live/exited status and log size, `/ps tail <pid> [lines]` to read their output, and `/ps kill <pid>` to stop them — even from a later session, since detached children outlive the AP process. Logs live in `<dataDir>/logs/` and are pruned after 7 days.
- **`/share`** (also `ap share [id]`): exports the transcript as **one self-contained HTML file** (inline CSS, zero external assets) under `<dataDir>/shares/` — collapsible tool calls/results, dark theme. No hosted service, no account: host it, mail it, attach it to a PR.
- **`AGENTS.md`** project notes supported alongside `AP.md`/`HARNESS.md`; `ap resume` (interactive picker) and `ap sessions search <q>` (ripgrep over transcripts).
- **Memory & self-improvement**: the agent persists what it learns to `<dataDir>/memory/*.md` — corrections you make **and** techniques that cracked tricky problems — and consults them in every future session (snapshotted per session, so prompt caching is unaffected). Recurring patterns get promoted into custom commands, closing the observe → persist → reuse loop natively at near-zero token cost. Want the full ceremony (episodic logs, confidence scoring)? `ap skills add charon-fan/agent-playbook --skill self-improving-agent`.

## Loop mode — run until verifiably done

`ap loop -p "goal"` keeps working until the goal is **verified** complete, not merely claimed complete:

```sh
ap loop -p "make all tests pass" --check "bun test"
ap loop -p "port utils/ to TS" --check "bun x tsc --noEmit" --check "bun test" --max 20
```

Each iteration: **work turn** → **objective gates** (every `--check` command must exit 0 — failures are fed back with their output, costing zero model tokens) → **auditor turn** (re-verifies the original goal with fresh read-only tool calls; outputs `LOOP_DONE` or a gap list that becomes the next work order). Engineered to be cheap to run indefinitely: one append-only session so provider prefix caching hits every iteration, fixed-byte verifier prompt, a **stall detector** (identical audits with no file changes twice → exit 2 instead of burning tokens forever), auto-**compaction** (near the context budget the session is summarized into a fresh one), and a checkpoint per mutating iteration (`/undo`-able trail). After **every iteration** a collapsed per-file diff is printed (`file  +added -removed`, computed from the shadow-git checkpoints — zero extra I/O on the hot path), plus a cumulative `total` when the loop ends. Exit codes: `0` verified done · `2` stalled or `--max` reached · `3` goal not applicable (`LOOP_BLOCKED`) · `130` ctrl+c.

## Skills (skills.sh / Claude Code compatible)

Skills are reusable instruction folders — `SKILL.md` with `name`/`description` frontmatter ([format docs](https://www.skills.sh/docs)). AP discovers them from `<project>/.ap/skills/`, `<project>/.claude/skills/`, `<dataDir>/skills/`, and `~/.claude/skills/` (so skills you already installed for Claude Code just work). Each skill costs **one line** in the system prompt; the agent reads the full SKILL.md only when a task matches (progressive disclosure — prompts stay small and cacheable).

```sh
ap skills                                        # list installed
ap skills add vercel-labs/agent-skills           # install every skill in a repo
ap skills add mattpocock/skills --skill loop-me  # install one
ap skills add https://www.skills.sh/owner/repo/name   # skills.sh URLs work too
ap skills remove <name>
```

The installer is zero-dep: GitHub tree API + raw downloads — no git, no npx. `/skills` lists them in the REPL. Skills are a full-profile feature; `--light` never injects them.

## MCP — the existing plug-in ecosystem, unchanged

AP is an [MCP](https://modelcontextprotocol.io) client, so any of the thousands of existing MCP servers (GitHub, Postgres, Slack, browsers, …) plug straight in — their tools appear to the model automatically as `mcp_<server>_<tool>`. Zero dependencies here too: MCP is just JSON-RPC 2.0, spoken over **stdio** (`Bun.spawn`, newline-delimited) or **Streamable HTTP** (plain `fetch`).

Config is the **same JSON as Claude Code** — paste a server's install snippet and it works. Either a `.mcp.json` in the project root or an `mcpServers` block in `ap.config.json` / `<dataDir>/config.json`:

```jsonc
{
  "mcpServers": {
    "filesystem": { "command": "bun", "args": ["x", "@modelcontextprotocol/server-filesystem", "."] },
    "remote":     { "url": "https://example.com/mcp", "headers": { "authorization": "Bearer …" } }
  }
}
```

```sh
ap mcp                                  # list servers, status, tools (exit 1 if any down — CI-friendly)
ap mcp call <server> <tool> '<json>'    # call one tool directly, no LLM (testing)
ap mcp add fs bun x @modelcontextprotocol/server-filesystem .   # add to global config
ap mcp add <name> --url <url> [--project]                       # HTTP server / write .mcp.json
ap mcp remove <name>
```

Engineering notes: servers connect **lazily before the first turn** (never on the startup path — startup stays ~50ms), the tool list is **frozen per process in a fixed order** so the request's schema bytes stay stable and provider prefix caching keeps hitting, a dead server degrades to a one-line warning (never a crash), tool results are capped at 40KB, and tools annotated `readOnlyHint` work in plan mode and run in parallel. Models that misname tools are tolerated (`server.tool`, `server__tool`, bare `tool` all resolve). MCP is a full-profile feature; `--light` never connects. `/mcp` lists servers in the REPL.

## Sessions (no database)

Each session is one append-only JSONL file in `<dataDir>/sessions/<id>.jsonl` — a `meta` line, then one line per message, flushed on every append. Crash-safe (torn last line ignored), resumable (`-c`, `--resume <id>`, `/resume`), greppable, and deleted by deleting the file.

## Zed / editors — ACP (Agent Client Protocol)

`ap acp` speaks [ACP](https://agentclientprotocol.com) v1 over stdio, so AP runs **inside Zed** (and any ACP editor) as a first-class agent. It's the same event stream, rendered differently: text → message chunks, reasoning → thought chunks, tool calls → live tool-call cards with the right icons, **sandbox permission requests → native editor dialogs**, plan/code → ACP session modes (switchable from the editor's mode picker), cancel → clean turn abort, and sessions persist (`loadSession`). MCP servers configured in the editor are passed straight through to AP's own MCP client.

**Slash commands work in the editor too**: AP advertises `/plan` · `/code` · `/model` · `/undo` · `/diff` · `/checkpoints` · `/mcp` · `/skills` · `/context` over ACP — type `/` in the agent panel and they autocomplete; they execute instantly inside the adapter (no model round-trip).

Zed `settings.json` (one `agent_servers` block — merge with existing entries, duplicate keys silently override):

```json
{ "agent_servers": { "AP": { "type": "custom", "command": "ap", "args": ["acp"] } } }
```

Then open the Agent Panel and pick **AP**. Provider/model flags carry through: `"args": ["acp", "--provider", "openrouter", "-m", "deepseek-v4-pro"]`. To debug a connection, Zed's `dev: open acp logs` (command palette) shows the raw JSON-RPC traffic; AP's diagnostics appear there prefixed `[acp]`.

## Server API (for programmatic drivers)

- `GET /health` → `{ok, version, provider, model}`
- `POST /session {cwd?, title?}` → `{id}`
- `POST /session/:id/message {text, system?, model?, response_format?}` → blocks → `{text, messages}`
- `GET /event` · `GET /session/:id/events` → SSE of AgentEvents (`text`/`reasoning`/`tool_start`/`tool_end`/`turn_end`/`done`/`error`, each with `sessionId`)
- `GET /session/:id/messages` → raw message array
- `DELETE /session/:id`

## Security

AP executes model-chosen commands on your machine by design — treat it like handing a very fast intern a terminal. The layered guardrails (write **and read** scoped sandbox with interactive permits, hard-denied AP-private data, dangerous-command blocklist, plan mode's structurally read-only schema set, required-argument schema validation, timeouts, output caps, 7-day background-log retention) plus the full threat model and vulnerability-reporting process are documented in [SECURITY.md](SECURITY.md). The honest boundary: this is a guardrail, not a VM — run genuinely untrusted code in a container. Zero runtime dependencies, and npm releases are published from GitHub Actions with **provenance attestation**.

## Why it's fast

- **Stable prompt prefix**: byte-identical system prompt + fixed-order terse tool schemas (~1K tokens total) → provider automatic prefix caching hits from turn 2 (measured ~85–90% of prompt tokens cached).
- **Streaming always**; tool_call deltas assembled by index; reasoning (`<think>`/`reasoning_content`) separated from answers and never re-sent as history.
- **Parallel reads**: multiple read/glob/grep calls in one turn run concurrently.
- **ripgrep everywhere**: grep *and* glob use `rg`, so ignored dirs are pruned during traversal, not filtered after.
- **Hard output caps**: read 50KB · bash 30KB middle-out · grep 200 matches · glob 300 files · 40KB backstop per result.
- **No permission stalls** by default (`yolo`), no database, lazy per-mode imports, `bun build --compile` single binary.

## Releasing

```sh
bun run push
```

One command: bumps the version, commits, tags `v0.x.y`, and pushes. The tag triggers GitHub Actions, which cross-compiles all four platform binaries (`ap-windows-x64.exe`, `ap-linux-x64`, `ap-darwin-arm64`, `ap-darwin-x64`) and attaches them to the release; the install one-liners always fetch the latest release. (Manual equivalent: `git tag v0.x.y && git push --tags`.)

**Registry installs** (after `npm publish` — package is `@sanjaydoppalapudi/agentplatform`, command is still `ap`):

```sh
npm i -g @sanjaydoppalapudi/agentplatform
bun add -g @sanjaydoppalapudi/agentplatform
yarn global add @sanjaydoppalapudi/agentplatform
pnpm add -g @sanjaydoppalapudi/agentplatform
```

## Credits

AP's code is original and dependency-free, but its best ideas were learned from opencode, Claude Code, Codex CLI, Cline, the MCP/ACP protocols, models.dev, and the skills.sh ecosystem — the full ledger of who inspired what is in [CREDITS.md](CREDITS.md).
