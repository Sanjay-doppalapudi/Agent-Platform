// Agent Skills (SKILL.md) — discovery, prompt injection, zero-dep installer.
// Compatible with the skills.sh / Claude Code format: a skill is a folder
// containing SKILL.md with `name:` + `description:` frontmatter; the body is
// loaded on demand (progressive disclosure — only one line per skill goes
// into the system prompt, the model reads the file when it needs it).
// Full profile only; --light never sees skills.
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
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

function scanDir(dir: string, source: string, into: Map<string, SkillInfo>) {
  let entries: string[];
  try { entries = readdirSync(dir); } catch { return; }
  for (const e of entries) {
    const skillMd = join(dir, e, "SKILL.md");
    if (!existsSync(skillMd)) continue;
    try {
      const raw = readFileSync(skillMd, "utf8");
      if (/^\s*internal:\s*true/m.test(raw.match(FRONT_RE)?.[1] ?? "")) continue;
      const fm = parseFrontmatter(raw);
      const name = fm["name"] || e;
      if (into.has(name)) continue; // earlier sources win (project > global)
      into.set(name, { name, description: fm["description"] ?? "", path: skillMd, source });
    } catch {}
  }
}

/** All available skills: project .ap/skills > .claude/skills > <dataDir>/skills > ~/.claude/skills. */
export function discoverSkills(config: Config): SkillInfo[] {
  const found = new Map<string, SkillInfo>();
  scanDir(join(config.cwd, ".ap", "skills"), "project", found);
  scanDir(join(config.cwd, ".claude", "skills"), "project", found);
  scanDir(join(config.dataDir, "skills"), "global", found);
  scanDir(join(homedir(), ".claude", "skills"), "claude", found);
  return [...found.values()].sort((a, b) => a.name.localeCompare(b.name));
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
  const tree = await ghJson(`https://api.github.com/repos/${owner}/${repo}/git/trees/HEAD?recursive=1`);
  const entries: TreeEntry[] = tree.tree ?? [];
  const prefix = pick.dir === "" ? "" : pick.dir + "/";
  const files = entries.filter(
    (e) => e.type === "blob" && (prefix === "" ? !e.path.includes("/") : e.path.startsWith(prefix)),
  );
  if (files.length > MAX_FILES_PER_SKILL) {
    throw new Error(`skill "${pick.name}" has ${files.length} files (cap ${MAX_FILES_PER_SKILL}) — install manually`);
  }
  const destRoot = join(dataDir, "skills", pick.name);
  for (const f of files) {
    if ((f.size ?? 0) > MAX_FILE_BYTES) { log(`  skip ${f.path} (too large)`); continue; }
    const rel = prefix === "" ? f.path : f.path.slice(prefix.length);
    const raw = await fetch(`https://raw.githubusercontent.com/${owner}/${repo}/HEAD/${f.path}`, {
      headers: { "user-agent": "ap-agent/1.0" },
      signal: AbortSignal.timeout(30_000),
    });
    if (!raw.ok) throw new Error(`download failed (${raw.status}): ${f.path}`);
    const dest = join(destRoot, rel);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, Buffer.from(await raw.arrayBuffer()));
    log(`  + ${rel}`);
  }
  return destRoot;
}

export function removeSkill(dataDir: string, name: string): boolean {
  const dir = join(dataDir, "skills", name);
  if (!existsSync(join(dir, "SKILL.md"))) return false;
  rmSync(dir, { recursive: true, force: true });
  return true;
}
