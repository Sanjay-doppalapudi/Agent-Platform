// Named agent profiles — opencode/Claude Code-style agent definition files.
// A profile is a markdown file whose frontmatter sets name/description/model/
// tools and whose BODY becomes the agent's role instructions (appended to the
// base system prompt). Discovery: .ap/agents > .claude/agents (compat) >
// <dataDir>/agents — first name wins. Used by `ap run --agent <name>` and the
// agent tool's {name} argument (subagents inherit the profile). Full profile
// only; one line per agent goes into the system prompt (snapshotted).
import { readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";
import type { Config } from "./config.ts";

export interface AgentDef {
  name: string;
  description: string;
  /** "model" or "provider/model" — overrides the session model for this agent. */
  model?: string;
  /** Canonical tool names this agent may use (undefined = profile default). */
  tools?: string[];
  /** Markdown body — the agent's role instructions. */
  body: string;
  path: string;
  source: string; // "project" | "claude" | "global"
}

const FRONT_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

function parseAgentFile(path: string, source: string): AgentDef | null {
  let raw: string;
  // Strip a UTF-8 BOM before matching frontmatter: with one present the `---`
  // opener never matches, so the whole block is silently ignored — a
  // tools:-restricted agent would quietly run with the FULL mutating toolset.
  try { raw = readFileSync(path, "utf8").replace(/^﻿/, ""); } catch { return null; }
  const m = raw.match(FRONT_RE);
  const fm: Record<string, string> = {};
  if (m) {
    for (const line of m[1]!.split(/\r?\n/)) {
      const kv = line.match(/^(\w[\w-]*):\s*(.+)$/);
      if (kv) fm[kv[1]!.toLowerCase()] = kv[2]!.trim().replace(/^["']|["']$/g, "");
    }
  }
  const body = (m ? raw.slice(m[0].length) : raw).trim();
  if (!body) return null;
  const name = (fm["name"] ?? basename(path, ".md")).toLowerCase();
  return {
    name,
    description: fm["description"] ?? "",
    model: fm["model"],
    tools: fm["tools"] ? fm["tools"].split(/[,\s]+/).map((t) => t.trim()).filter(Boolean) : undefined,
    body,
    path,
    source,
  };
}

function scanDir(dir: string, source: string, into: Map<string, AgentDef>) {
  let entries: string[];
  try { entries = readdirSync(dir); } catch { return; }
  for (const f of entries) {
    if (!f.endsWith(".md")) continue;
    const def = parseAgentFile(join(dir, f), source);
    if (def && !into.has(def.name)) into.set(def.name, def);
  }
}

/** All defined agents: .ap/agents > .claude/agents > <dataDir>/agents. */
export function discoverAgents(config: Config): AgentDef[] {
  const found = new Map<string, AgentDef>();
  scanDir(join(config.cwd, ".ap", "agents"), "project", found);
  scanDir(join(config.cwd, ".claude", "agents"), "claude", found);
  scanDir(join(config.dataDir, "agents"), "global", found);
  return [...found.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function getAgent(config: Config, name: string): AgentDef | null {
  return discoverAgents(config).find((a) => a.name === name.toLowerCase()) ?? null;
}

const MAX_PROMPT_AGENTS = 20;

/** One line per agent for the system prompt (progressive disclosure). */
export function agentsPromptBlock(agents: AgentDef[]): string {
  if (!agents.length) return "";
  const lines = agents
    .slice(0, MAX_PROMPT_AGENTS)
    .map((a) => `- ${a.name}: ${(a.description || a.body.replace(/\s+/g, " ")).slice(0, 150)}`);
  return `\n\nNamed agents — specialist profiles for delegation. Pass name to the agent tool (e.g. {"name": "${agents[0]!.name}", "task": "..."}) to run a subtask as that specialist:\n${lines.join("\n")}`;
}
