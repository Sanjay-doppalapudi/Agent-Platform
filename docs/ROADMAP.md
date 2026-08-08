# AP feature roadmap — competitive survey & implementation report

Generated 2026-08-01, status updated 2026-08-08. Sources surveyed: opencode, Claude Code, OpenAI Codex CLI, Cline, xAI Grok Build, Nous Hermes Agent, OpenClaw, tmux. Constraint for everything below: **zero or near-zero dependencies, nothing on the startup hot path, prompt-prefix byte-stability preserved, every feature gated off in `ap --light`.**

## Status (2026-08-08) — shipped since this survey

- **All of P0 and P1** (checkpoints/undo, subagents, custom commands, AGENTS.md, @file, tool hooks, /compact, session search, worktrees, fetch+todo tools, resume picker).
- **Web**: `websearch` (DuckDuckGo scrape) + plain `fetch` (SSRF hardening: metadata/link-local + IPv6-embedded; DNS pin + redirect re-check); browser rendering remains disabled.
- **Loop mode** (`ap loop`): work→check→audit until verifiably done; stall detection, shared Compaction 2.0, per-iteration diffs, LOOP_BLOCKED.
- **Read-scoped sandbox**: reads outside the workspace permit-gated, AP-private data hard-denied, bash path scanning; compound bash permission segments.
- **Skills**: skills.sh / Claude Code SKILL.md packs, zero-dep GitHub installer (nested folders); `/skills reload`.
- **MCP client** (was P2 → shipped): stdio + Streamable HTTP, Claude-Code-format config, dynamic tools frozen for cache stability, `ap mcp` CLI; `/mcp reload`, soft auto-background onto task queue.
- **ACP adapter** (was P2 → shipped): `ap acp` for Zed — session modes, native permission dialogs, slash commands (incl. `/commit`/`/pr`), session load, editor-MCP passthrough.
- **Lifecycle hooks**: `hooks.onDone`/`onError` — shell command or webhook POST; `preCompact`/`postCompact`; afterEdit `AP_ARGS.paths`.
- **Tier A (2026-08)**: repo-keyed memory, Compaction 2.0 (`/archives`, `/restore-context`, auto-memory), compound bash permissions, `repomap` tool, in-process agent channels.
- **Tier B (2026-08)**: `/flow` list/last, `/thinking`, `/confirm edits`, `.ap/DECISIONS.md`, `/rewind`, `/commit --staged|--sign`, bracketed paste, richer Retry-After, REPL `/agent` profiles, `/steer`.
- **Git + tmux (2026-08)**: `git.autoBranch` on first mutation, `/pr` + `ap pr` (`gh pr create`), optional `ap tmux` / `/spawn` (unix; clear fallback on Windows).
- **Windows tool fixes**: spawn `rg` via `Bun.which` absolute path; grep file-path targets must not use the file as spawn cwd.

Remaining candidates: none from the original Tier A/B/remaining set — further work is polish and competitive catch-up only. See [README.md](../README.md) for the user-facing feature list.

## The two-profile model (implemented)

- **`ap --light`** — the frozen fast profile: smallest system prompt (no memory injection, no plans note), no plan HTML export, compact banner. This is the contract: every future feature must check `config.light` and cost nothing there.
- **`ap`** — the full harness, which grows by this roadmap.

---

## 1. What each tool does well (survey findings)

| Tool | Standout features worth studying |
|---|---|
| **opencode** | Plan/build agents, `@general` subagent for searches, Tab agent-switching, models.dev provider catalog (we already share this), session sharing, desktop shell |
| **Claude Code** | Subagents, hooks (pre/post tool shell commands), custom slash commands from markdown files, CLAUDE.md project memory, plan mode, checkpoints/rewind, headless mode, plugins, GitHub @-mention integration |
| **Codex CLI** | OS-level sandboxing (Seatbelt/Landlock), approval modes ladder, AGENTS.md convention, `codex exec` non-interactive, session resume, profiles |
| **Cline** | Plan/Act split, workspace checkpoints with visual diff review, monitors linter/compiler errors and self-fixes, auto-approve toggles, `.clinerules`, MCP marketplace, coordinator→specialist multi-agent |
| **Grok Build** | Full TUI with mouse/scrollback, ACP (Agent Client Protocol) for editor embedding, checkpoints, skills+hooks+plugins, Rust modular split of tui/runtime/tools |
| **Hermes Agent** | Self-improving skill loop (creates skills from experience), FTS session search with LLM summarization for cross-session recall, seven execution backends (incl. Vercel Sandbox), messaging gateways, cron scheduler |
| **OpenClaw** | Gateway architecture (one control plane, many channels), local-first data, pairing/approval security for remote channels, capability marketplace |

## 2. Feature verdicts for AP

Legend: **P0** build next · **P1** valuable, after P0 · **P2** someday · **Skip** conflicts with AP's identity (speed, zero-dep, terminal-first).

### P0 — high value, near-zero weight

| Feature | Inspired by | Design (zero-dep) | Est. size |
|---|---|---|---|
| **Git checkpoints + `/undo`** | Claude Code rewind, Cline checkpoints, Grok Build | Shadow repo trick: `git --git-dir=<dataDir>/checkpoints/<session> --work-tree=<cwd>` — commit after every mutating turn without touching the user's real git. `/undo` = checkout previous checkpoint; `/diff` = diff vs last checkpoint. Pure `git` CLI (already on every dev machine). | ~120 lines |
| **Subagents** | opencode @general, Cline coordinators, Hermes | Spawn our own binary: `ap run -p "<subtask>" --json --light` as a child process, stream its NDJSON into a collapsed UI line, feed final text back as a tool result. New `agent` tool (schema added only in full profile — light keeps 6 tools). Parallel subagents = parallel spawns. | ~100 lines |
| **Custom slash commands** | Claude Code, opencode | `.ap/commands/<name>.md` in repo or dataDir → `/name args` expands the file as the user message (`$ARGS` substitution). File-based, discovered lazily when `/` menu opens. | ~40 lines |
| **AGENTS.md support** | Codex convention | Read `AGENTS.md` alongside `AP.md`/`HARNESS.md` in prompt.ts (first found wins). Industry-standard file name = free interop. | ~3 lines |
| **`@file` mentions** | Cline | In REPL input, `@src/foo.ts` inlines a capped read of that file into the message (with Tab-completion from a cheap `rg --files` in the slash-menu infra). Saves a model round-trip per referenced file. | ~60 lines |
| **Post-edit diagnostics hook** | Cline's linter monitoring | Config `hooks: { afterEdit: "bun x tsc --noEmit --pretty false" }` — run after mutating turns, append failures as a tool-style message so the model self-fixes. Generalizes to any linter. This is 80% of LSP's value at 0% of its weight. | ~50 lines |

### P1 — valuable, moderate effort

| Feature | Inspired by | Design | Est. size |
|---|---|---|---|
| **`/compact`** (summarize-and-continue) | Claude Code | Ask the current model to summarize history → start fresh context with summary + last N messages. Manual command first; auto-trigger at budget later. | ~60 lines |
| **Session search** | Hermes FTS recall | `ap sessions search <query>` — rg over the JSONL session dir (we already ship rg). LLM summarization skip — plain matches are enough. | ~40 lines |
| **Hooks (pre/post tool)** | Claude Code, Grok Build | Config map `hooks: {preBash: "cmd", postWrite: "cmd"}` — shell out, nonzero exit blocks the tool with stderr as the error. | ~50 lines |
| **Worktree-per-task** (see §4) | Claude Code worktrees | `/worktree <task>`: `git worktree add <dataDir>/worktrees/<slug> -b ap/<slug>`, switch session cwd there; `/worktree merge` when done. | ~80 lines |
| **`fetch` tool** (URL → text) | Grok Build web search | Plain `fetch()` + HTML tag-strip + 50KB cap, added to full-profile schemas only. No search API (needs keys); URL fetch alone unlocks docs lookups. | ~50 lines |
| **Todo tool** | Claude Code TaskCreate | In-memory per-session todo list tool + dim checklist render — keeps long multi-step tasks on rails. | ~60 lines |
| **Resume picker** | Codex | `ap resume` = interactive session list using the existing slash-menu renderer. | ~30 lines |

### P2 — someday / conditional

- **MCP client**: SHIPPED (mcp.ts) — stdio + Streamable HTTP, zero-dep, lazy connect before the first turn, tool list frozen per process for prompt-cache stability, `.mcp.json`/`mcpServers` Claude Code config compat, `ap mcp list/call/add/remove`.
- **ACP editor embedding**: SHIPPED (acp.ts) — ACP v1 for Zed: modes, permission dialogs, slash commands, `session/load`, editor-MCP passthrough.
- **Session sharing** (opencode): reuse the plan-HTML exporter to render a whole session transcript to a self-contained HTML file (`/share` → file, user hosts it however they like). ~40 lines, could promote to P1.
- **Cron/scheduled runs**: CUT (decided 2026-08): no scheduler will be built — the OS scheduler (Task Scheduler / cron) invoking `ap run -p … --json` is the supported pattern.
- **Skill self-improvement loop** (Hermes, charon-fan/agent-playbook): SHIPPED natively (2026-08) — memory now captures successful techniques as well as corrections, with promotion of recurring patterns into custom commands baked into the prompt; the full playbook remains available as a skills.sh install for those who want it (see CREDITS.md).

### Skip (with reasons)

- **Full-screen TUI with mouse** (Grok Build, opencode): alternate-screen buffers, layout engines, mouse protocols — this is where CLIs get heavy and where we deliberately went inline-scroll. Ctrl+O re-render already covers the main need.
- **Messaging gateways / personal-assistant channels** (OpenClaw, Hermes): a different product category. AP ships an **in-process** agent channel bus for coordinating subagents/workflows inside one repo session — not a multi-surface gateway.
- **Marketplaces / plugin SDKs**: ecosystem plays that need mass; file-based commands+hooks deliver the useful subset.
- **OS-level sandboxing** (Codex Seatbelt/Landlock): the right way needs platform-native code (Windows has no Landlock equivalent accessible from Bun). Our jail+danger-block+permit model is the honest zero-dep ceiling; document Vercel Sandbox/Docker as the escalation path for untrusted work.
- **Browser automation** (Cline): needs Playwright/CDP — heavyweight. The `fetch` tool covers read-only web needs.

## 3. tmux — shipped as an optional unix adapter

tmux = terminal multiplexer: named **sessions** that survive disconnect (detach/attach), **windows/panes**, and a scripting surface (`new-session -d`, `send-keys`, `capture-pane`).

**SHIPPED** (`src/tmux.ts`): detected via `Bun.which("tmux")` — never a hard dependency.

1. **Parallel background agents**: `/spawn <task>` → detached `ap run --json --light` in `tmux new-session -d -s ap-<slug>`; `tmux attach -t ap-<slug>` to watch; `/tmux capture <session>` pulls pane text.
2. **Layout bootstrap**: `ap tmux` / `/tmux layout` opens panes for ap | shell | spare.
3. **Persistence**: long `ap run` jobs on remote Linux survive SSH drops.

**Windows-native fallback (shipped):** background `bash` + `/ps` + worktrees + `--resume`. On native Windows, `ap tmux` / `/spawn` print a clear hint (use WSL or the fallback).

## 4. Git workflows: worktrees, branches, checkpoints — shipped

Three layers, all pure `git` CLI, no libraries:

1. **Checkpoint layer (safety, invisible)** — shadow repo per session. Auto-commit after each mutating turn: `/undo`/`/diff`/`/restore` without polluting real history.
2. **Branch layer (hygiene)** — `git.autoBranch: true`: first mutation on a protected branch → `git switch -c ap/<slug>`. `/commit [--staged|--sign]`: stage + commit with model-drafted message + approval. `/pr` + `ap pr`: `gh pr create` helper (never force-pushes; `--yes` required on CLI).
3. **Worktree layer (parallelism)** — `/worktree new|list|back|merge` with `ap/<slug>` branches; pairs with tmux `/spawn` for isolated parallel agents.

## 5. Suggested build order

1. ~~**P0 batch**~~ DONE.
2. ~~Worktrees · auto-branch · `/commit` · `/pr`~~ DONE.
3. ~~`/compact` · session search · fetch · todo · `/ps`~~ DONE; Compaction 2.0 DONE.
4. ~~tmux adapter · `/share`~~ DONE.
5. ~~MCP · ACP~~ DONE.

Every step: typecheck (local `tsc` preferred) → `bun run smoke` / `ap tool` live verification → `bun run push` (version bump, binaries, npm — automated).
