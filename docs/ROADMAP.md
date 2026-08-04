# AP feature roadmap — competitive survey & implementation report

Generated 2026-08-01, status updated 2026-08-04. Sources surveyed: opencode, Claude Code, OpenAI Codex CLI, Cline, xAI Grok Build, Nous Hermes Agent, OpenClaw, tmux. Constraint for everything below: **zero or near-zero dependencies, nothing on the startup hot path, prompt-prefix byte-stability preserved, every feature gated off in `ap --light`.**

## Status (2026-08-04) — shipped since this survey

- **All of P0 and P1** (checkpoints/undo, subagents, custom commands, AGENTS.md, @file, tool hooks, /compact, session search, worktrees, fetch+todo tools, resume picker).
- **Web**: `websearch` (DuckDuckGo scrape) + `fetch render:true` (system Chrome/Edge headless).
- **Loop mode** (`ap loop`): work→check→audit until verifiably done; stall detection, compaction, per-iteration diffs, LOOP_BLOCKED.
- **Read-scoped sandbox**: reads outside the workspace permit-gated, AP-private data hard-denied, bash path scanning.
- **Skills**: skills.sh / Claude Code SKILL.md packs, zero-dep GitHub installer.
- **MCP client** (was P2 → shipped): stdio + Streamable HTTP, Claude-Code-format config, dynamic tools frozen for cache stability, `ap mcp` CLI.
- **ACP adapter** (was P2 → shipped): `ap acp` for Zed — session modes, native permission dialogs, slash commands, session load, editor-MCP passthrough.
- **Lifecycle hooks**: `hooks.onDone`/`onError` — shell command or webhook POST when a turn finishes.

Remaining candidates: `/share` transcript export, tmux adapter (unix), `/ps` background manager, auto-branch + `/commit`, skill self-improvement.

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
- **Skill self-improvement loop** (Hermes): our memory system is the seed; a "promote memory → command template" step could come later.

### Skip (with reasons)

- **Full-screen TUI with mouse** (Grok Build, opencode): alternate-screen buffers, layout engines, mouse protocols — this is where CLIs get heavy and where we deliberately went inline-scroll. Ctrl+O re-render already covers the main need.
- **Messaging gateways / channels** (OpenClaw, Hermes): a different product category (personal assistant). AP is a repo tool.
- **Marketplaces / plugin SDKs**: ecosystem plays that need mass; file-based commands+hooks deliver the useful subset.
- **OS-level sandboxing** (Codex Seatbelt/Landlock): the right way needs platform-native code (Windows has no Landlock equivalent accessible from Bun). Our jail+danger-block+permit model is the honest zero-dep ceiling; document Vercel Sandbox/Docker as the escalation path for untrusted work.
- **Browser automation** (Cline): needs Playwright/CDP — heavyweight. The `fetch` tool covers read-only web needs.

## 3. tmux — what it's for and how AP uses it

tmux = terminal multiplexer: named **sessions** that survive disconnect (detach/attach), **windows/panes**, and a scripting surface (`new-session -d`, `send-keys`, `capture-pane`) that makes it an orchestration substrate.

**Reality check for this machine:** tmux does not run on native Windows (WSL/MSYS2 only). So AP treats tmux as an *optional adapter*, detected via `Bun.which("tmux")`:

1. **Parallel background agents** (the killer use): `/spawn <task>` →
   `tmux new-session -d -s ap-<slug> 'ap run -p "<task>" --cwd <worktree>'` — the agent keeps running after you close the terminal; `tmux attach -t ap-<slug>` to watch live; `capture-pane -p` to pull output back into the parent session. Pairs perfectly with worktree-per-task (§4).
2. **Layout bootstrap**: `ap tmux` opens a session with pane 1 = `ap`, pane 2 = dev server logs, pane 3 = shell.
3. **Persistence**: long `ap run` jobs on a remote Linux box survive SSH drops.

**Windows-native fallback (already 80% built):** background `bash` tool (detached + log file) + `ap serve` sessions + `--resume` give the same survive-and-reattach properties; a `/ps` command listing background pids + tailing their logs closes the gap. Verdict: implement the fallback pieces first (they work everywhere), add the ~60-line tmux adapter for unix hosts.

## 4. Git workflows: worktrees, branches, checkpoints

Three layers, all pure `git` CLI, no libraries:

1. **Checkpoint layer (safety, invisible)** — shadow repo per session (`--git-dir` under dataDir, work-tree = workspace). Auto-commit after each mutating turn: free unlimited `/undo`/`/diff`/`/restore` without polluting the user's history, works even in non-git folders. This is how Claude Code's rewind and Cline's checkpoints behave.
2. **Branch layer (hygiene)** — config `git.autoBranch: true`: first mutation on a clean default branch → `git switch -c ap/<session-slug>` (never commit to main uninvited). `/commit` command: stage + commit with a model-drafted message shown for approval.
3. **Worktree layer (parallelism)** — `/worktree <task>` creates `git worktree add <dir> -b ap/<slug>`; each subagent or tmux background agent gets its own worktree so parallel tasks never collide on the working tree; `/worktree merge` rebases/merges back and removes it. This is the pattern Claude Code power users run, and it composes with §3.1 into: *one command → isolated branch+worktree+background agent → review diff → merge*.

PR flow: `gh pr create` via the bash tool already works today (user's gh is authed); a `/pr` command is a one-liner template over it.

## 5. Suggested build order

1. ~~**P0 batch** (checkpoints+/undo, subagents, custom commands, AGENTS.md, @file, post-edit hook)~~ DONE.
2. Worktrees DONE; auto-branch + `/commit` (§4) remain.
3. ~~`/compact`, session search, `fetch` tool, todo tool~~ DONE; `/ps` background manager remains.
4. tmux adapter (unix), `/share` transcript export — still open.
5. ~~Re-evaluate MCP once real demand appears~~ DONE — MCP client + ACP adapter shipped (see Status).

Every step: typecheck → `ap tool`/live verification → `bun run push` (version bump, binaries, npm — automated).
