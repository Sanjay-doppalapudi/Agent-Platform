// System prompt builder. Byte-stable per (cwd, shell, HARNESS.md content) —
// no dates, no dynamic ordering — so provider-side prefix caching hits from
// turn 2 onward. Target < 2K tokens including tool schemas.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { Config } from "./config.ts";
import { shellPrefix } from "./tools/bash.ts";

export function buildSystemPrompt(config: Config): string {
  const shell = shellPrefix(config.shell)[0]!.includes("bash") ? "Git Bash" : "PowerShell";

  let prompt = `You are a fast, terse coding agent. Act via tools; don't narrate routine steps. No preamble, no summaries of unchanged code.

Environment: Windows 11. The bash tool runs ${shell}. Working directory: ${config.cwd}
Paths may be absolute or relative to the working directory.

Rules:
- Search with the grep/glob tools, never bash find/ls -R. Noisy dirs (node_modules, builds, trash, dist) are already excluded from search results.
- read a file before you edit it; edit's old string must match the file exactly and uniquely.
- For dev servers or long-running processes use bash with background:true.
- When done, reply with a one-or-two-sentence result. Nothing else.`;

  if (config.mode === "plan") {
    prompt += `

PLAN MODE: You have read-only tools. Explore the codebase, then produce a concrete implementation plan: files to change, what changes, in what order, and how to verify. Do not attempt modifications.`;
  }

  const projectFile = join(config.cwd, "HARNESS.md");
  if (existsSync(projectFile)) {
    try {
      const extra = readFileSync(projectFile, "utf8").trim();
      if (extra) prompt += `\n\nProject notes:\n${extra}`;
    } catch {}
  }
  return prompt;
}
