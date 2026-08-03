// System prompt builder. Byte-stable per (cwd, shell, HARNESS.md content) —
// no dates, no dynamic ordering — so provider-side prefix caching hits from
// turn 2 onward. Target < 2K tokens including tool schemas.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Config } from "./config.ts";
import { shellPrefix } from "./tools/bash.ts";

const MEMORY_CHAR_CAP = 2000;

/** Concatenated saved memories, capped. */
function readMemories(memDir: string): string {
  let out = "";
  try {
    for (const f of readdirSync(memDir).sort()) {
      if (!f.endsWith(".md")) continue;
      out += readFileSync(join(memDir, f), "utf8").trim() + "\n---\n";
      if (out.length > MEMORY_CHAR_CAP) {
        return out.slice(0, MEMORY_CHAR_CAP) + "\n[more memories truncated]";
      }
    }
  } catch {}
  return out;
}

// Memories are SNAPSHOTTED once per session: a memory written mid-session is
// already in the conversation history, so deferring its injection to the next
// session loses nothing — and the system prompt stays byte-identical for the
// whole session, so the provider's prefix cache is never invalidated.
const memorySnapshots = new Map<string, string>();

function memoriesForSession(memDir: string, sessionKey: string): string {
  const key = `${memDir}|${sessionKey}`;
  let snap = memorySnapshots.get(key);
  if (snap === undefined) {
    snap = readMemories(memDir);
    memorySnapshots.set(key, snap);
  }
  return snap;
}

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

  if (config.sandbox === "workspace") {
    prompt += `
- Writes outside the workspace need user permission; dangerous shell commands are blocked automatically — prefer working inside the workspace.`;
  }

  if (config.mode === "plan") {
    prompt += `

PLAN MODE: You have read-only tools. Explore the codebase, then produce a concrete implementation plan: files to change, what changes, in what order, and how to verify. Do not attempt modifications.`;
  }

  const memDir = join(config.dataDir, "memory");
  prompt += `

Memory: when the user corrects you or wants something different from what you did, save it — write ${memDir}\\<short-slug>.md with exactly three lines: "Title: …", "User wanted: …", "Why (guess): …". Consult the saved memories below before repeating a choice the user disliked.`;
  const memories = memoriesForSession(memDir, config.sessionId ?? "");
  if (memories) prompt += `\n\nSaved user preferences:\n${memories}`;

  if (config.sessionId) {
    const plansDir = join(tmpdir(), ".ap", config.sessionId);
    prompt += `\n\nPlans from this session are saved as HTML in ${plansDir} — read earlier ones from there when useful. Folders of other sessions are strictly off-limits.`;
  }

  for (const name of ["AP.md", "HARNESS.md"]) {
    const projectFile = join(config.cwd, name);
    if (existsSync(projectFile)) {
      try {
        const extra = readFileSync(projectFile, "utf8").trim();
        if (extra) prompt += `\n\nProject notes:\n${extra}`;
      } catch {}
      break;
    }
  }
  return prompt;
}
