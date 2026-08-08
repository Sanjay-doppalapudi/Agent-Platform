# Credits

AP is original code with zero runtime dependencies, but very few of its *ideas*
are original — they were learned from the projects below. Credit where it's due.

## Products that shaped the feature set

- **[opencode](https://github.com/anomalyco/opencode)** ([opencode.ai](https://opencode.ai)) —
  the direct benchmark. Named agent profiles, per-tool allow/ask/deny permissions,
  auto-compaction, session sharing, the plan/build split, and using models.dev as
  the provider catalog are all patterns AP deliberately matched (with a
  lighter-weight, zero-dependency stance).
- **[Claude Code](https://claude.com/claude-code)** (Anthropic) — the file formats
  and conventions AP speaks for compatibility: `.mcp.json`, `.claude/skills`,
  `.claude/agents`, project-notes files (`CLAUDE.md`-style `AP.md`), markdown
  custom slash commands, pre/post tool hooks, and the checkpoints/rewind idea.
- **[OpenAI Codex CLI](https://github.com/openai/codex)** — the `AGENTS.md`
  convention and the approval-modes way of thinking about agent safety.
- **[Cline](https://github.com/cline/cline)** — the plan/act split, workspace
  checkpoints with diff review, and the monitor-diagnostics-and-self-fix idea
  behind AP's `afterEdit` hook.
- **[Nous Hermes Agent](https://github.com/NousResearch)** — session-search
  recall and the skill self-improvement loop framing.
- **[xAI Grok Build](https://x.ai)** — precedent for ACP editor embedding from a
  fast terminal agent.

## Protocols and services

- **[Model Context Protocol](https://modelcontextprotocol.io)** (Anthropic) — the
  open plug-in standard AP's `mcp.ts` client speaks.
- **[Agent Client Protocol](https://agentclientprotocol.com)** (Zed Industries) —
  the editor protocol behind `ap acp`.
- **[models.dev](https://models.dev)** — the open provider/model catalog powering
  `/models`, pricing on the status line, and `/effort`'s reasoning-support check.
- **DuckDuckGo** — the HTML endpoint behind the key-free `websearch` tool.

## Skills ecosystem

- **[skills.sh](https://www.skills.sh)** / the Agent Skills `SKILL.md` format —
  the packaging AP's skill discovery and installer are compatible with.
- **[mattpocock/skills — loop-me](https://www.skills.sh/mattpocock/skills/loop-me)** —
  the seed idea for `ap loop`'s work→verify-until-done cycle (AP's version adds
  objective check gates, an auditor turn, stall detection, and auto-compaction).
- **[charon-fan/agent-playbook — self-improving-agent](https://www.skills.sh/charon-fan/agent-playbook/self-improving-agent)** —
  the observe→extract→persist→validate self-improvement loop that inspired
  extending AP's native memory to capture successful techniques (not just
  corrections) and promote recurring ones into custom commands. AP adopts the
  idea natively rather than as an always-on skill to keep prompts small; the
  full playbook installs with
  `ap skills add charon-fan/agent-playbook --skill self-improving-agent`.

## Foundations

- **[Bun](https://bun.sh)** (Oven) — runtime, bundler, and single-binary compiler.
- **[ripgrep](https://github.com/BurntSushi/ripgrep)** (Andrew Gallant) — the
  search engine behind `grep`/`glob`/`repomap`'s speed.
- **[GitHub CLI (`gh`)](https://cli.github.com)** — used by `/pr` and `ap pr`
  when present; never bundled.
- **tmux** — optional unix adapter (`ap tmux` / `/spawn`); detected on PATH,
  never a dependency. Windows falls back to `/ps` + worktrees.
- **Vercel Sandbox** — the guardrail-not-a-VM framing that shaped AP's sandbox
  honesty (documented limits, containers recommended for untrusted code).
