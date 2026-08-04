# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Agent Platform (`ap`) — a minimal coding-agent CLI optimized for raw speed. Bun + TypeScript, **zero runtime dependencies** (Bun built-ins only: fetch, Bun.serve, Bun.spawn, node:readline). Speaks the OpenAI-compatible chat-completions wire format to any provider (OpenCode Go, OpenRouter, NVIDIA NIM, arbitrary base_url). npm package: `@sanjaydoppalapudi/agentplatform`, bin name `ap`.

## Commands

```sh
bun run dev              # run from source (bun run src/index.ts)
bun x tsc --noEmit       # typecheck — run after every change; there are no unit tests
bun run compile          # build ap.exe + prewarm run (Defender scans happen at build time, not first launch)
bun run dist             # cross-compile all four platform binaries into dist/
bun run src/index.ts tool grep '{"pattern":"foo"}'   # exercise one tool directly, no LLM call
bun run src/index.ts run -p "task" --cwd <dir>       # one-shot agent run (needs a provider key)
```

Releases: `git tag v0.x.y && git push --tags` → `.github/workflows/release.yml` cross-compiles and attaches `dist/*` to the GitHub release, which `install.ps1` / `install.sh` download.

On Windows, `bun build --compile` cannot overwrite a **running** ap.exe (EPERM) — rename it aside first (`Rename-Item ap.exe ap.exe.old-<stamp>`), then build.

## Architecture

One process, one event stream. `agent.ts` runs the loop and emits `AgentEvent`s (`text`/`reasoning`/`tool_start`/`tool_end`/`turn_end`/`done`/`error`); every front-end — REPL (`repl.ts`), one-shot/NDJSON (`run.ts`), HTTP+SSE server (`server.ts`) — is just a different renderer of that same stream. To add a capability, emit an event; never print from inside the loop or tools.

Data flow per turn: `repl/run/server` → `runTurn()` (agent.ts) → `streamChat()` (provider.ts, plain fetch + retry/backoff) → `consumeSSE()` (stream.ts, assembles text + index-keyed tool_call deltas, strips `<think>`/`reasoning_content` into the reasoning channel) → tool dispatch (tools/index.ts registry) → results appended as `role:"tool"` messages → repeat until no tool calls.

- **Modes** (`config.mode`): `code` sends the full tool-schema set for the profile; `plan` sends only the read-only subset so mutation is structurally impossible, plus a backstop block in agent.ts for hallucinated calls.
- **Web access** (full profile only): `websearch` scrapes DuckDuckGo's HTML endpoint with plain fetch (no API key); `fetch` with `render:true` spawns the system's installed Chrome/Edge headless (`--headless=new --dump-dom`, in tools/fetch.ts) — never bundle or download a browser; degrade to plain fetch when none is found.
- **Sessions** (`session.ts`): append-only JSONL per session under `<dataDir>/sessions/`, no database. A torn final line is skipped on load — this is the crash-safety mechanism; never buffer writes.
- **Server** (`server.ts`): endpoint shapes deliberately mirror what 5Pages' `build-server/lib/opencode-runtime.js` calls on opencode (session create → blocking message → SSE events) so it can replace opencode there with a thin adapter.
- **Credentials** (`creds.ts`): keys live in `<dataDir>/credentials.json`, ACL-locked to the current user. On Windows, icacls grants must use fully-qualified `DOMAIN\user` — a bare username silently grants to nobody when the machine name matches the username.
- **Sandbox is read-scoped too** (`tools/shared.ts`): `readRoots` = workspace + dataDir skills/memory/commands + session plans + `~/.claude/skills`; reads elsewhere permit-gated (read/grep/glob AND bash via `scanCmdPaths` path-token scan incl. `..` escapes). `isPrivatePath` (dataDir sessions/checkpoints/credentials.json/config.json) is hard-denied for read AND write — permits cannot override. Loop mode exits 3 on a `LOOP_BLOCKED:` reply instead of letting the model wander when a goal doesn't apply.
- **models.dev** (`models.ts`): provider/model catalog, fetched lazily and disk-cached 24h. Never loaded on the startup path.
- **Loop mode** (`loop.ts`): `ap loop -p goal [--check cmd]…` — work turn → objective check commands (exit-0 gates, fed back on failure) → auditor turn (fixed-byte VERIFY_PROMPT, read-only steering, `LOOP_DONE` sentinel, confirmation passes when the audit used mutating tools). Stall detector (2 identical audits + no mutations → exit 2) and auto-compaction (summary → fresh session past 60% of contextBudgetChars) keep unlimited looping affordable.
- **Skills** (`skills.ts`): skills.sh/Claude-format SKILL.md packs. Discovery order `.ap/skills` > `.claude/skills` > `<dataDir>/skills` > `~/.claude/skills`; one line per skill in the system prompt (snapshotted per session), body read on demand. Installer uses the GitHub tree API + raw.githubusercontent — never git/npx. Full profile only.
- **MCP client** (`mcp.ts`): zero-dep JSON-RPC 2.0 over stdio (Bun.spawn, newline-delimited; Windows `.cmd` shims run under `cmd /c`) or Streamable HTTP (fetch, `mcp-session-id` tracked). Config = Claude Code's format: project `.mcp.json` (walked upward, wins) merged with `mcpServers` in ap.config.json/`<dataDir>/config.json`. Front-ends `await initMcp()` before the FIRST turn (REPL kicks it off in the background at startup) — never on the startup path. Tools register once per process as dynamic tools (`registerDynamicTools` in tools/index.ts) named `mcp_<server>_<tool>`, sorted, then frozen → schema bytes stay stable for caching. `readOnlyHint === true` ⇒ readOnly (plan-mode visible, parallel-safe); everything else is treated as mutating, and the plan-mode backstop in agent.ts blocks it. Dead servers warn + skip. Full profile only; subagents run `--light` so they never spawn servers.

**The `--light` profile is frozen**: core six tools, smallest prompt, no memory/skills/plans/subagents/checkpoints — new features must never leak into it (gate on `config.light`).

## Invariants that make it fast — do not break

- **Byte-stable prompt prefix.** The system prompt (`prompt.ts`) contains no timestamps or dynamic ordering, tool schemas are in a FIXED registry order (`tools/index.ts`), and messages are append-only. This is what makes provider-side automatic prefix caching hit (~85–90% of prompt tokens). Anything that varies per request must go after the history, never into the system prompt or tool list.
- **Zero npm runtime deps and lazy per-mode `import()`** in `index.ts` — startup is ~45ms compiled; adding a dependency or a top-level import of a mode is a regression.
- **Every tool output is capped** (read 50KB / bash 30KB middle-out / grep 200 matches / glob 300 files / 40KB backstop in `execTool`). New tools must self-cap.
- **grep AND glob shell out to ripgrep** so `HARD_IGNORES` dirs (node_modules, builds, trash, dist*, …) are pruned during traversal — the target repo (5Pages) has ~5,000 generated files that make post-filtering unusable.
- **Reasoning is never re-sent**: `<think>` blocks and `reasoning_content` are surfaced to the UI but excluded from stored history.
- Terminal rendering is line-buffered markdown (`md.ts`) and a raw-mode line reader (`input.ts`); the readLine `prompt` string must be single-line — it is re-rendered on every keypress.

## Legacy-name compatibility (post-rebrand)

The tool was renamed harness → Agent Platform. Keep accepting the old names: data dir `~/.harness` (new: `~/.ap`), project config `harness.config.json` (new: `ap.config.json`), project notes `HARNESS.md` (new: `AP.md`) — 5Pages still uses `HARNESS.md`. Env vars are still `HARNESS_PROVIDER`/`HARNESS_MODEL`/`HARNESS_BASE_URL`/`HARNESS_API_KEY`.

## Alias tolerance

Weaker models misname tool arguments. `edit` accepts `old`/`old_string`/`oldText` and `new`/`new_string`/`newText` (`tools/edit.ts`), and `ui.ts` renderDiff mirrors the aliases. Extend this pattern rather than letting a turn fail on argument naming.
