<p align="center"><img src="logo.svg" width="200" alt="Agent Platform"></p>

# Agent Platform (`ap`)

Minimal, fast coding agent CLI. OpenAI-compatible providers only (OpenCode Go, OpenRouter, NVIDIA NIM, any base_url). Core tools: `read` · `write` · `edit` · `bash` · `glob` · `grep` (+ `agent` · `fetch` · `todo` · `websearch` · `repomap` · `artifact` in the full profile). Plugs into the existing ecosystem: **MCP client** (any MCP server's tools become agent tools), **ACP agent** (runs inside Zed), **skills.sh / Claude Code skills**, lifecycle **hooks/webhooks**, optional **tmux** adapter, and real-git helpers (`/commit`, `/pr`, `git.autoBranch`). Zero runtime dependencies — Bun built-ins only. ~45ms cold start.

```
◆ AP · opencode-go/minimax-m3
  cwd C:\projects\my-app
  session 20260801...-ab3d · type / for commands · ! for shell · ctrl+o details · ctrl+c abort

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

Run **`ap doctor`** any time something misbehaves: it checks ripgrep, git, the shell, data-dir writability, the selected provider, whether an API key actually resolves, credential-file permissions, endpoint reachability, and MCP server health — printing a concrete fix for every failure (exit 1 when something is broken, so CI can gate on it). `ap doctor --offline` skips the network and MCP probes.

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
ap flow <name> [args…]  # run a user workflow (.ap/workflows/<name>.ts)
ap watch                # live view of running ap sessions, agents, flows
ap skills               # list / install SKILL.md packs (skills.sh compatible)
ap mcp                  # connect MCP servers — their tools become agent tools
ap acp                  # ACP agent for editors (Zed): stdio, modes, permissions
ap serve [--port 4141]  # HTTP server mode (sessions + SSE)
ap models [query]       # search the models.dev catalog (context + pricing)
ap auth <provider>      # store an API key
ap trust [accept|…]     # trust this workspace's ap.config.json / .mcp.json (interactive)
ap resume / ap sessions # pick a session to resume / list + full-text search
ap share [id]           # export a transcript as one self-contained HTML file
ap ps [tail|kill]       # background processes from bash background:true
ap pr [--yes]           # create a GitHub PR via gh (preview without --yes)
ap tmux [layout|…]      # optional unix tmux adapter (graceful on Windows)
ap doctor               # diagnose the environment (deps, keys, endpoint, MCP) — exit 1 if broken
ap prompt [--cwd dir]   # print the system prompt used for a directory
ap tool grep '{"pattern":"foo"}'   # run one tool directly (testing)
bun run smoke           # real user-path smoke (no LLM key for most checks)
ap help <command>       # detailed help (run · loop · pr · tmux · skills · mcp · …)
```

Common flags: `--provider <name>` · `-m/--model <id>` · `--cwd <dir>` · `--mode plan|code` (`--plan`) · `--light` · `--effort <level>` · `--agent <name>` · `--session <id>` · `-c/--continue` · `--base-url <url> --api-key <key>` for any ad-hoc endpoint.

Optional router policies keep the normal provider path intact while selecting explicit targets and falling back only on transient failures before output:

```json
"router": { "targets": ["opencode-go/minimax-m3", "openrouter/anthropic/claude-sonnet-4.5"], "fallback": true }
```

Targets use each provider's endpoint from models.dev unless `providers` config overrides it. Credentials stay separate per provider via `ap auth <provider>` (or the provider's env var), and new models.dev providers/models can be selected without adding a provider block. Authentication and model errors never fall through. Only OpenAI Chat Completions-compatible providers auto-route; add custom protocol support explicitly if needed.

### REPL

Type `/` to open the command menu (↑/↓ navigate, Enter/Tab select, Esc close; ↑/↓ recall history otherwise). A line starting with **`!`** is a **shell escape** — the rest runs in your own shell (like `!` in psql/gdb) instead of going to the model, streaming output straight through; it's your terminal, so it bypasses the agent guardrails.

| Command | Effect |
|---|---|
| `/plan` / `/code` | switch mode — plan is structurally read-only and produces an implementation plan; code (default) has all tools. Optional `planModel` / `codeModel` swap the live provider on switch |
| `/model` | interactive picker: every provider (models.dev + config) → models with context/pricing; or `/model <provider>/<model>` |
| `/theme [name]` | list/switch themes (`default` · `mono` · `nord` · `dracula` · `gruvbox` · `solarized` · `matrix`); saved to config; `NO_COLOR` forces mono |
| `/effort` · `/thinking` | reasoning effort (`low`/`medium`/`high`/`off`) · show/hide streamed thinking |
| `/confirm edits on\|off` | force ask before every `edit`/`write` even under `permissions: yolo` |
| `/new` `/resume` `/sessions` `/rename` | session management (list / delete / rename) |
| `/steer [text]` | queue coaching for the next turn (or mid-turn via ctrl+s) |
| `/undo` `/diff` `/checkpoints` `/restore` | shadow-git checkpoint ops |
| `/worktree` `/commit` `/pr` | real-git worktrees · commit (`--staged`/`--sign`) · open a GitHub PR via `gh` |
| `/spawn` `/tmux` `/ps` | detach in tmux (unix) · tmux list/layout/capture · background bash processes |
| `/tasks` `/flow` `/artifacts` `/watch` | background subagent tasks · workflows (`list`/`last`/`<name>`) · artifacts · live process viewer |
| `/compact` `/archives` `/restore-context` `/rewind` | Compaction 2.0 — summarize, list archives, reinject an archive note, drop last N user turns |
| `/agent` `/agents` | apply/clear a named agent profile · list subagents this session |
| `/mcp` `/skills` `/sandbox` | MCP status/`reload` · skills/`reload` · sandbox state/toggle (`workspace`/`container`/`off`) |
| `/share` `/system` `/context` | export HTML transcript · inspect system prompt · token usage |
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
| `permission` | — | per-tool rules, evaluated first: `{"fetch": "deny", "edit": "ask", "mcp_*": "ask", "bash": {"git push*": "ask", "*": "allow"}}` — tool keys and bash command patterns take `*` globs; **compound bash** (`cmd1 && cmd2`, pipes, `;`) takes the **strictest** segment verdict; `deny` blocks outright, `ask` uses the interactive permit (auto-denied headless), `allow` skips only the ask gate (sandbox + bashGuard still apply) |
| `sandbox` | `"workspace"` | `"workspace"`: writes/edits outside the workspace (+ data dir + session plans) need a y/N/always permission (headless denies unless `--allow-outside`). `"container"` (`--sandbox container`): each `bash` runs in a throwaway docker/podman container — only the workspace mounted, egress off by default — a **real** OS boundary. `"off"` / `--no-sandbox`: no file gates |
| `sandboxImage` | `"alpine"` | image for `sandbox:"container"` (the command runs as `/bin/sh -c`, workspace at `/workspace`) — set to match your stack (`node:20`, …) |
| `network` | `"allow"` | egress policy for `fetch`/`websearch`/`bash` URL tokens: `"deny"` blocks all, a `string[]` is a suffix-match hostname allowlist. Cloud-metadata / link-local hosts are blocked under **every** policy. Best-effort in `workspace` mode; OS-enforced (`--network none`) in `container` mode |
| `bashGuard` | `"on"` | dangerous shell patterns (recursive absolute deletes, format, registry edits, curl\|bash, force-push incl. `git.exe`, …) are auto-blocked, warned, and logged to `<dataDir>/blocked-commands.jsonl` for provider feedback |
| `streamIdleSeconds` | 90 | stalled provider streams abort and retry once instead of hanging (0 = off) |
| `maxIterations` | 40 | agent loop guard |
| `contextBudgetChars` | 400000 | old tool results elided beyond this |
| `redactEnv` | true | `.env` values read back as `KEY=***` |
| `shell` | `"auto"` | Git Bash if found, else PowerShell (`bash`/`powershell`/`cmd`) |
| `parallelPolicy` | `"safe"` | read-only tools run parallel, mutations serial (`all`/`none`) |
| `ignore` | `[]` | extends the hard ignore list (node_modules, .git, dist*, builds, trash, uploads, …) |
| `checkpoints` | `"on"` | shadow-git checkpoint after every mutating turn (`"off"` disables) |
| `autoCompact` | `"on"` | REPL auto-summarizes into a fresh session at 85% of the context budget (`"off"` disables; `/compact` stays manual) |
| `autoMemory` | `"on"` | after compact, extract 0–3 memory cards into the repo-keyed memory dir |
| `mcpAutoBackgroundMs` | `30000` | soft ms before a blocking MCP tool call backgrounds onto the task queue (`0` = never) |
| `showReasoning` | `"on"` | show streamed thinking in the compact REPL (`/thinking off` toggles) |
| `confirmEdits` | — | `true` / `/confirm edits on` — ask before every edit/write even under yolo |
| `planModel` / `codeModel` | — | optional `"provider/model"` (or bare model id) swapped in on `/plan` / `/code` |
| `git.autoBranch` | `false` | on first mutating tool call while on a protected branch, create and switch to `ap/<slug>` |
| `theme` | `"default"` | color theme: `default`/`mono`/`nord`/`dracula`/`gruvbox`/`solarized`/`matrix` (also `/theme`) |
| `reasoningEffort` | — | `"low"`/`"medium"`/`"high"` — default `reasoning_effort` sent with every request (`/effort` and `--effort` override) |
| `hooks` | — | `preBash`/`preWrite`/`preEdit`/`afterEdit` tool hooks (+ `AP_ARGS.paths` for afterEdit) · `preCompact`/`postCompact` · `onDone`/`onError` lifecycle (command or webhook URL) |
| `mcpServers` | — | MCP servers, Claude Code format (also read from a project `.mcp.json`) |

**API keys** live apart from config in `<dataDir>/credentials.json`, file-ACL'd to your user (`ap auth <provider>`). Resolution: `--api-key` → `HARNESS_API_KEY` → config `apiKey` → provider env var → credential store.

**Per-project prompt notes:** drop an `AP.md` (or legacy `HARNESS.md`) / `AGENTS.md` in a project root — its content is appended to the system prompt for that project. Optional `.ap/DECISIONS.md` is also injected (architecture decisions the agent should respect) when present.

## Sandbox

**Reads are scoped**: the agent can freely read the workspace plus the skills/memory/commands dirs and its session plans folder. Reading anywhere else — other projects, your home dir, `ls`/`cat` via bash included (best-effort path scan, `..` escapes counted, **symlink targets resolved**) — triggers the interactive `[y/N/a=always]` permission and is denied headlessly unless `--allow-outside`.

**AP-private data is never accessible** — hard-denied for read *and* write even if you'd approve (only `sandbox:"off"` lifts it): session transcripts, checkpoints, `credentials.json`, `config.json`, and the workspace-trust store.

**Mutations are jailed** the same way: `write`/`edit` outside the workspace/data-dir/plans folder need permission. Files that grant **code execution on the next run** are hard-denied outright — `ap.config.json` / `harness.config.json` / `.mcp.json` (in any directory), `.git/hooks/`, `.ap|.claude/commands/` — as are the prompt-note files (`AP.md`/`AGENTS.md`/`.ap/skills|agents|workflows`/`DECISIONS.md`). On Windows, trailing dots/spaces are normalized so `ap.config.json.` can't dodge those name checks.

**Dangerous bash patterns are auto-blocked** (never prompted) with a ⚠ warning and a JSONL log entry you can share with your provider.

**The honest boundary**: `sandbox:"workspace"` is a guardrail, not a VM — file-path containment holds well, but `bash` is host code execution, so the pattern scan is best-effort and a determined model can escape it (e.g. a path decoded at runtime). For genuinely untrusted code, **`sandbox:"container"`** (`--sandbox container`) runs every `bash` inside a throwaway docker/podman container with only the workspace mounted and egress off by default — a real OS boundary. `/sandbox` shows state (writable + readable roots) and toggles per-session.

**Workspace trust.** A project's `ap.config.json` and `.mcp.json` can execute code (hooks, MCP servers) and relax guardrails, so they take effect only once you trust the workspace. Until then an untrusted project gets a safe allowlist (`theme`/`ignore`/…) — its `hooks`/`mcpServers`/`sandbox`/`network`/`permission`/`provider` overrides, project skills/agents/workflows/commands, and `.mcp.json` are ignored, with a warning listing what was stripped. Run **`ap trust accept`** (interactive only — it names the resolved git root before you confirm; the agent can never grant trust for itself). `ap trust status|revoke|list` manage it.

Weaker-model tolerance: tool names (`search`→grep, `create`→write, …) and argument names (`file_path`, `command`, `query`, …) are alias-normalized, malformed JSON args get auto-repaired, and edits retry with CRLF and trailing-whitespace normalization — each recovery saves a full model round-trip.

## Full-profile features (absent in `--light`)

- **Checkpoints**: every mutating turn auto-commits the workspace to a shadow git repo (your real git history is untouched; works in non-git folders). `/undo`, `/diff [n|git|<branch>]`, `/checkpoints`, `/restore <hash>`.
- **Subagents**: the `agent` tool delegates independent subtasks to parallel child processes (`ap run --light` under the hood — children can't recurse). Live nested `↳` progress lines stream in, and `/agents` lists every subagent with status, steps, and duration. Add `"background": true` to **detach** the task: the tool returns immediately, the subagent runs on, and its result is folded into your next message as a `<task-result>` note. `/tasks` lists them (with per-task steps/duration), `/tasks kill <id>` stops one; an audit trail lands in `<dataDir>/tasks.jsonl`. Optional `channel` on the agent tool posts/reads an in-process note bus (`channels.ts`) so siblings can coordinate without mid-turn history injection.
- **Dynamic workflows**: write a `.ap/workflows/<name>.ts` that `export default async function ({ agent, parallel, log, args, channel })` — deterministic control flow around bounded `agent()` calls. Pass a JSON schema (`agent(task, { schema })`) to force validated JSON with an automatic one-shot retry. Run with `ap flow <name> [args…]` or `/flow` (`list` / `last` / `<name>`). A model can't launch a flow — you do.
- **Artifacts**: the `artifact` tool writes a self-contained HTML page to `<dataDir>/artifacts/` with a no-network CSP. `/artifacts` lists; `ap serve` exposes `GET /artifacts/<file>.html`.
- **Named agents**: `.ap/agents/<name>.md` (frontmatter `description`/`model`/`tools`; body = role). Delegate with `{"name": "reviewer", "task": …}`, run headlessly with `ap run --agent <name>`, or `/agent <name>` / `/agent clear` for the live REPL session.
- **Custom slash commands**: `.ap/commands/<name>.md` → `/name args` with `$ARGS` substitution.
- **`@file` mentions**: `@src/foo.ts` inlines the file (8KB cap). Bracketed paste is supported in the REPL input.
- **Hooks**: `hooks.preBash/preWrite/preEdit` (nonzero exit blocks) and `hooks.afterEdit` (e.g. `bun x tsc --noEmit` — failures fed back, max 2 rounds/turn; `AP_ARGS.paths` lists touched files). Compaction hooks: `preCompact`/`postCompact`. Lifecycle: `hooks.onDone` / `hooks.onError` — shell (`AP_EVENT`/`AP_PAYLOAD`) or `http(s)://` JSON POST.
- **Web + outline tools**: `websearch` (DuckDuckGo HTML, no key) · `fetch` (HTTP(S) text, 50KB; cloud-metadata / link-local / IPv6-embedded SSRF blocked; `render:true` disabled) · `todo` · **`repomap`** (ripgrep outline of defs under a path — full profile only). `websearch` and `fetch` both honour the `network` egress policy (`deny` / allowlist).
- **Real git**:
  - **`git.autoBranch: true`**: first mutating tool call on a protected branch → create/switch to `ap/<slug>` (never touches main uninvited).
  - **`/commit [--staged] [--sign] [message]`**: stage + commit to your real history with a drafted message and `y/N` confirm — **never pushes**. Protected branches offer an `ap/<slug>` switch first.
  - **`/pr` / `ap pr`**: draft a GitHub PR via `gh` (`--draft`/`--base`/`--title`/`--body`; CLI requires `--yes` or `AP_PR_YES=1` to actually create).
  - **`/worktree new|list|back|merge`**: isolated worktree + `ap/<slug>` branch per task.
- **Compaction 2.0**: `/compact` summarizes into a fresh session (also auto at 85% budget). Archives land under `<dataDir>/archives/` with `index.jsonl` + `parentSessionId`; `/archives` lists them; `/restore-context <id>` injects a note from an archive. `/rewind [N]` drops the last N user turns from history (not files). With `autoMemory: "on"`, compact extracts 0–3 cards into **repo-keyed memory** (`memory/git-<hash>/`, keyed by `git rev-parse --git-common-dir`).
- **Memory & self-improvement**: persists corrections and successful techniques under the repo-keyed memory dir and consults them in future sessions (snapshotted per session for prompt-cache stability). Recurring patterns promote into custom commands. Want the full ceremony (episodic logs, confidence scoring)? `ap skills add charon-fan/agent-playbook --skill self-improving-agent`.
- **Steer**: `/steer <text>` (or ctrl+s mid-turn) queues coaching that lands on the next user message — never mid-history.
- **`/ps`** (also `ap ps`): background processes from `bash background:true` — list / tail / kill across sessions.
- **tmux** (optional, unix): `ap tmux` / `/tmux` / `/spawn <task>` — layout, list, capture, detach an `ap run --light`. Native Windows prints a clear fallback (`/worktree` + `/ps` or WSL).
- **`/share`** (also `ap share [id]`): one self-contained HTML transcript under `<dataDir>/shares/`.
- **`AGENTS.md`** + `.ap/DECISIONS.md`; `ap resume` picker; `ap sessions search <q>`; `ap watch` live process viewer.
- **MCP soft auto-background**: long MCP tools (~30s by default, `mcpAutoBackgroundMs`) move onto the task queue instead of blocking the turn forever; `/mcp reload` and `/skills reload` rebuild discovery mid-session.

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

Engineering notes: servers connect **lazily before the first turn** (never on the startup path — startup stays ~50ms), the tool list is **frozen per process in a fixed order** so the request's schema bytes stay stable and provider prefix caching keeps hitting, a dead server degrades to a one-line warning (never a crash), tool results are capped at 40KB, tools annotated `readOnlyHint` work in plan mode and run in parallel, and long calls soft-auto-background onto the task queue (`mcpAutoBackgroundMs`, default 30s). Models that misname tools are tolerated (`server.tool`, `server__tool`, bare `tool` all resolve). MCP is a full-profile feature; `--light` never connects. `/mcp` lists servers; `/mcp reload` reconnects and rebuilds tools (accepts a prefix-cache miss).

## Sessions (no database)

Each session is one append-only JSONL file in `<dataDir>/sessions/<id>.jsonl` — a `meta` line, then one line per message, flushed on every append. Crash-safe (torn last line ignored), resumable (`-c`, `--resume <id>`, `/resume`), greppable, and deleted by deleting the file.

## Zed / editors — ACP (Agent Client Protocol)

`ap acp` speaks [ACP](https://agentclientprotocol.com) v1 over stdio, so AP runs **inside Zed** (and any ACP editor) as a first-class agent. It's the same event stream, rendered differently: text → message chunks, reasoning → thought chunks, tool calls → live tool-call cards with the right icons, **sandbox permission requests → native editor dialogs**, plan/code → ACP session modes (switchable from the editor's mode picker), cancel → clean turn abort, and sessions persist (`loadSession`). MCP servers configured in the editor are passed straight through to AP's own MCP client.

**Slash commands work in the editor too**: AP advertises `/plan` · `/code` · `/model` · `/undo` · `/diff` · `/checkpoints` · `/commit` · `/pr` · `/mcp` · `/skills` · `/context` over ACP — type `/` in the agent panel and they autocomplete; they execute instantly inside the adapter (no model round-trip).

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

AP executes model-chosen commands on your machine by design — treat it like handing a very fast intern a terminal. The layered guardrails (write **and read** scoped sandbox with interactive permits and resolved symlink targets, hard-denied AP-private data + privileged config + trust store, workspace trust for project config/MCP, dangerous-command blocklist, a configurable `network` egress policy, plan mode's structurally read-only schema set, required-argument schema validation, timeouts, output caps, 7-day background-log retention) plus the full threat model and vulnerability-reporting process are documented in [SECURITY.md](SECURITY.md). The honest boundary: `sandbox:"workspace"` is a guardrail, not a VM — for genuinely untrusted code use the built-in **`sandbox:"container"`** mode (bash in docker/podman, workspace-only mount, egress off). Zero runtime dependencies, and npm releases are published from GitHub Actions with **provenance attestation**.

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
