// System prompt builder. Byte-stable per (cwd, shell, HARNESS.md content) —
// no dates, no dynamic ordering — so provider-side prefix caching hits from
// turn 2 onward. Target < 2K tokens including tool schemas.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Config } from "./config.ts";
import { shellPrefix } from "./tools/bash.ts";
import { agentsPromptBlock, discoverAgents } from "./agents.ts";
import { discoverSkills, skillsPromptBlock } from "./skills.ts";

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

// Skills are snapshotted per session for the same cache-stability reason.
const skillSnapshots = new Map<string, string>();

function skillsForSession(config: Config): string {
  const key = `${config.dataDir}|${config.cwd}|${config.sessionId ?? ""}`;
  let snap = skillSnapshots.get(key);
  if (snap === undefined) {
    snap = skillsPromptBlock(discoverSkills(config));
    skillSnapshots.set(key, snap);
  }
  return snap;
}

// Named agent profiles — same per-session snapshot pattern.
const agentSnapshots = new Map<string, string>();

function agentsForSession(config: Config): string {
  const key = `${config.dataDir}|${config.cwd}|${config.sessionId ?? ""}`;
  let snap = agentSnapshots.get(key);
  if (snap === undefined) {
    snap = agentsPromptBlock(discoverAgents(config));
    agentSnapshots.set(key, snap);
  }
  return snap;
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
  // Describe the ACTUAL platform. This used to hardcode "Windows 11" and
  // infer the shell from the substring "bash", so every macOS/Linux run told
  // the model it was on Windows (and /usr/bin/bash was labelled "Git Bash").
  // Still byte-stable per machine — nothing here varies between requests.
  const exe = shellPrefix(config.shell)[0]!;
  const isWin = process.platform === "win32";
  const os = isWin ? "Windows" : process.platform === "darwin" ? "macOS" : "Linux";
  const shell = isWin
    ? (exe.includes("bash") ? "Git Bash" : exe.includes("powershell") ? "PowerShell" : "cmd")
    : (exe.endsWith("bash") ? "bash" : "sh");

  let prompt = `You are a fast, terse coding agent. Act via tools; don't narrate routine steps. No preamble, no summaries of unchanged code.

Environment: ${os}. The bash tool runs ${shell}. Working directory: ${config.cwd}
Paths may be absolute or relative to the working directory.

Rules:
- Search with the grep/glob tools, never bash find/ls -R. Noisy dirs (node_modules, builds, trash, dist) are already excluded from search results.
- read a file before you edit it; edit's old string must match the file exactly and uniquely.
- For dev servers or long-running processes use bash with background:true.
- When done, reply with a one-or-two-sentence result. Nothing else.`;

  if (config.sandbox === "workspace") {
    prompt += `
- Stay inside the working directory. Reading or writing outside it requires user permission — never explore unrelated folders, other projects, or AP's own data (sessions, credentials, checkpoints). Dangerous shell commands are blocked automatically.`;
  }

  if (!config.light) {
    prompt += `
- Use the agent tool to delegate independent subtasks in parallel (background:true returns immediately and reports back on a later turn); use todo to track multi-step work; use websearch to find web pages and fetch to read them. Answer questions about current events by searching, never from memory.
- Use the artifact tool when the user wants a report, dashboard or diagram to look at: it saves one self-contained HTML page (inline CSS/JS only — remote resources are blocked).
- "Workflow" means an AP workflow, not a script in another language: write .ap/workflows/<name>.ts exporting \`export default async function ({ agent, parallel, log, args }) { … }\`, where agent(task, {schema?}) runs a subagent and parallel([...thunks]) runs them concurrently. The user runs it with \`ap flow <name>\` or /flow. Never hand-roll a runner in Python/Node for this.`;
  }

  if (config.mode === "plan") {
    prompt += `

PLAN MODE: You have read-only tools. Explore the codebase, then produce a concrete implementation plan: files to change, what changes, in what order, and how to verify. Do not attempt modifications.`;
  }

  if (!config.light) {
    const memDir = join(config.dataDir, "memory");
    prompt += `

Memory: when the user corrects you or wants something different from what you did, save it — write ${memDir}\\<short-slug>.md with exactly three lines: "Title: …", "User wanted: …", "Why (guess): …". Also save one after cracking a tricky problem whose approach will recur (Title = the technique). Consult the saved memories below before repeating a choice the user disliked; if one keeps applying, promote it into a custom command (.ap/commands/<name>.md) so it becomes reusable structure.`;
    const memories = memoriesForSession(memDir, config.sessionId ?? "");
    if (memories) prompt += `\n\nSaved user preferences:\n${memories}`;

    prompt += skillsForSession(config);
    prompt += agentsForSession(config);

    if (config.sessionId) {
      const plansDir = join(tmpdir(), ".ap", config.sessionId);
      prompt += `\n\nPlans from this session are saved as HTML in ${plansDir} — read earlier ones from there when useful. Folders of other sessions are strictly off-limits.`;
    }
  }

  for (const name of ["AP.md", "AGENTS.md", "HARNESS.md"]) {
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
