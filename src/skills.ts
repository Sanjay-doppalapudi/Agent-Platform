// Agent Skills (SKILL.md) — discovery, prompt injection, zero-dep installer.
// Compatible with the skills.sh / Claude Code format: a skill is a folder
// containing SKILL.md with `name:` + `description:` frontmatter; the body is
// loaded on demand (progressive disclosure — only one line per skill goes
// into the system prompt, the model reads the file when it needs it).
// Full profile only; --light never sees skills.
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import type { Config } from "./config.ts";

export interface SkillInfo {
  name: string;
  description: string;
  /** Absolute path to SKILL.md. */
  path: string;
  source: string; // "project" | "global" | "claude"
}

const FRONT_RE = /^---\r?\n([\s\S]*?)\r?\n---/;

function parseFrontmatter(md: string): Record<string, string> {
  const m = md.match(FRONT_RE);
  const out: Record<string, string> = {};
  if (!m) return out;
  for (const line of m[1]!.split(/\r?\n/)) {
    const kv = line.match(/^(\w[\w-]*):\s*(.+)$/);
    if (kv) out[kv[1]!.toLowerCase()] = kv[2]!.trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

function scanDir(dir: string, source: string, into: Map<string, SkillInfo>, depth = 0) {
  let entries;
  try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const leaf = e.name;
    const skillMd = join(dir, leaf, "SKILL.md");
    if (existsSync(skillMd)) {
      try {
        const raw = readFileSync(skillMd, "utf8").replace(/^﻿/, ""); // BOM voids frontmatter
        if (/^\s*internal:\s*true/m.test(raw.match(FRONT_RE)?.[1] ?? "")) continue;
        const fm = parseFrontmatter(raw);
        const name = fm["name"] || leaf;
        if (into.has(name)) continue; // earlier sources win (project > global)
        into.set(name, { name, description: fm["description"] ?? "", path: skillMd, source });
      } catch {}
    } else if (depth < 2) {
      // Nested category folders: skills/group/foo/SKILL.md (depth-capped)
      scanDir(join(dir, leaf), source, into, depth + 1);
    }
  }
}

// Process-lifetime cache (same rationale as agents.ts): discovery is walked
// from the system-prompt builder and from CLI list/install paths; rescanning
// every call was pure overhead once the session snapshot is fixed.
const skillCache = new Map<string, SkillInfo[]>();

/** All available skills: project .ap/skills > .claude/skills > <dataDir>/skills > ~/.claude/skills. */
export function discoverSkills(config: Config): SkillInfo[] {
  const key = JSON.stringify([config.dataDir, config.cwd]);
  const hit = skillCache.get(key);
  if (hit) return hit;
  const found = new Map<string, SkillInfo>();
  scanDir(join(config.cwd, ".ap", "skills"), "project", found);
  scanDir(join(config.cwd, ".claude", "skills"), "project", found);
  scanDir(join(config.dataDir, "skills"), "global", found);
  scanDir(join(homedir(), ".claude", "skills"), "claude", found);
  const list = [...found.values()].sort((a, b) => a.name.localeCompare(b.name));
  skillCache.set(key, list);
  return list;
}

/** Test helper / post-install refresh. */
export function clearSkillCache(): void {
  skillCache.clear();
}

const MAX_PROMPT_SKILLS = 30;

/** One line per skill for the system prompt (progressive disclosure). */
export function skillsPromptBlock(skills: SkillInfo[]): string {
  if (!skills.length) return "";
  const lines = skills
    .slice(0, MAX_PROMPT_SKILLS)
    .map((s) => `- ${s.name}: ${s.description.slice(0, 200)} → ${s.path}`);
  return `\n\nSkills — reusable instruction packs. When a task matches one, read its SKILL.md path FIRST and follow it:\n${lines.join("\n")}`;
}

// ---------------------------------------------------------------------------
// Installer: `ap skills add <owner>/<repo> [--skill name]` — GitHub only,
// plain fetch (git tree API + raw.githubusercontent), no git, no npx.
// ---------------------------------------------------------------------------

const MAX_FILES_PER_SKILL = 40;
const MAX_FILE_BYTES = 512 * 1024;

/**
 * Is this a safe relative path to write under the skill directory?
 *
 * Git tree paths are always '/'-separated, so a backslash is never legitimate
 * — and on Windows it IS a separator, which made `..\..\config.json` land on
 * the user's own config (whose hooks.onDone is an arbitrary shell command) or
 * credentials.json. This is the one writer that does not go through the tool
 * sandbox, so it validates its own paths.
 */
export function safeSkillPath(rel: string): boolean {
  if (!rel || rel.includes("\\") || rel.includes("\0")) return false;
  if (/^[A-Za-z]:/.test(rel) || rel.startsWith("/")) return false; // absolute
  return rel.split("/").every((seg) => seg !== "" && seg !== "." && seg !== ".." && !/[:<>"|?*]/.test(seg));
}

interface TreeEntry { path: string; type: string; size?: number }

async function ghJson(url: string): Promise<any> {
  const res = await fetch(url, {
    headers: { "user-agent": "ap-agent/1.0", accept: "application/vnd.github+json" },
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`GitHub API ${res.status} for ${url}`);
  return res.json();
}

/** Parse "owner/repo", a github.com URL, or a skills.sh URL into {owner, repo, skill?}. */
export function parseSource(src: string): { owner: string; repo: string; skill?: string } {
  let m = src.match(/^https?:\/\/(?:www\.)?skills\.sh\/([^/]+)\/([^/]+)(?:\/([^/?#]+))?/);
  if (m) return { owner: m[1]!, repo: m[2]!, skill: m[3] };
  m = src.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)/);
  if (m) return { owner: m[1]!, repo: m[2]!.replace(/\.git$/, "") };
  m = src.match(/^([\w.-]+)\/([\w.-]+)$/);
  if (m) return { owner: m[1]!, repo: m[2]! };
  throw new Error(`cannot parse source "${src}" — use owner/repo, a github.com URL, or a skills.sh URL`);
}

/** List installable skills in a repo: [{name, dir}] where dir contains SKILL.md. */
export async function listRemoteSkills(owner: string, repo: string): Promise<{ name: string; dir: string }[]> {
  const tree = await ghJson(`https://api.github.com/repos/${owner}/${repo}/git/trees/HEAD?recursive=1`);
  const entries: TreeEntry[] = tree.tree ?? [];
  const out: { name: string; dir: string }[] = [];
  for (const e of entries) {
    if (e.type !== "blob" || !/(^|\/)SKILL\.md$/.test(e.path)) continue;
    const dir = e.path === "SKILL.md" ? "" : dirname(e.path);
    const name = dir === "" ? repo : dir.split("/").pop()!;
    if (name.startsWith(".")) continue;
    out.push({ name, dir });
  }
  return out;
}

export async function installSkill(
  dataDir: string,
  owner: string,
  repo: string,
  pick: { name: string; dir: string },
  log: (line: string) => void,
): Promise<string> {
  try {
    const tree = await ghJson(`https://api.github.com/repos/${owner}/${repo}/git/trees/HEAD?recursive=1`);
    const entries: TreeEntry[] = tree.tree ?? [];
    const prefix = pick.dir === "" ? "" : pick.dir + "/";
    const files = entries.filter(
      (e) => e.type === "blob" && (prefix === "" ? !e.path.includes("/") : e.path.startsWith(prefix)),
    );
    if (files.length > MAX_FILES_PER_SKILL) {
      throw new Error(`skill "${pick.name}" has ${files.length} files (cap ${MAX_FILES_PER_SKILL}) — install manually`);
    }
    if (!safeSkillPath(pick.name)) throw new Error(`unsafe skill name from repo: ${pick.name}`);
    const destRoot = join(dataDir, "skills", pick.name);
    for (const f of files) {
      if ((f.size ?? 0) > MAX_FILE_BYTES) { log(`  skip ${f.path} (too large)`); continue; }
      const rel = prefix === "" ? f.path : f.path.slice(prefix.length);
      if (!safeSkillPath(rel)) { log(`  skip ${f.path} (unsafe path)`); continue; }
      const raw = await fetch(`https://raw.githubusercontent.com/${owner}/${repo}/HEAD/${f.path}`, {
        headers: { "user-agent": "ap-agent/1.0" },
        signal: AbortSignal.timeout(30_000),
      });
      if (!raw.ok) throw new Error(`download failed (${raw.status}): ${f.path}`);
      const dest = join(destRoot, rel);
      // Belt and braces: even if the validator ever misses a form, never write
      // outside the skill's own directory.
      if (!resolve(dest).startsWith(resolve(destRoot) + sep)) {
        log(`  skip ${f.path} (escapes the skill directory)`);
        continue;
      }
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, Buffer.from(await raw.arrayBuffer()));
      log(`  + ${rel}`);
    }
    return destRoot;
  } finally {
    clearSkillCache();
  }
}

export function removeSkill(dataDir: string, name: string): boolean {
  // Names are identifiers — never path segments. Without this check,
  // `ap skills remove ../credentials.json`-style input could traverse.
  if (!safeSkillPath(name) || name.includes("/")) return false;
  const root = resolve(dataDir, "skills");
  const dir = resolve(root, name);
  if (!dir.startsWith(root + sep) && dir !== root) return false;
  if (!existsSync(join(dir, "SKILL.md"))) return false;
  try {
    rmSync(dir, { recursive: true, force: true });
    return true;
  } finally {
    clearSkillCache();
  }
}
