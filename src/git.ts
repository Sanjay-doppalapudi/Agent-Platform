// Real-git helpers for /commit (distinct from checkpoint.ts, which drives the
// SHADOW repo). Everything here touches the user's actual history, so the
// rules are: never commit to a protected branch without an explicit switch,
// never push, and always show the message before committing. Pure string
// helpers are exported separately so they can be unit-tested without a repo.
import type { Config } from "./config.ts";

export interface GitState {
  repo: boolean;
  branch: string;
  protectedBranch: boolean;
  /** Files with staged or unstaged changes (porcelain, capped). */
  dirty: string[];
}

const PROTECTED = new Set(["main", "master", "trunk", "develop", "development", "prod", "production"]);

/** Branches AP must not commit to without the user opting in. */
export function isProtectedBranch(name: string): boolean {
  const n = name.trim().toLowerCase();
  return PROTECTED.has(n) || /^release[/-]/.test(n);
}

/** Task text → a safe `ap/<slug>` branch name (never empty, never too long). */
export function slugifyBranch(text: string, prefix = "ap/"): string {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .split("-")
    .filter(Boolean)
    .slice(0, 6)
    .join("-")
    .slice(0, 40)
    .replace(/-+$/, "");
  return `${prefix}${slug || "work"}`;
}

/**
 * Normalize a model-drafted commit message: strip code fences/quotes/leading
 * "Commit message:" preamble, drop any Co-Authored-By or tool-attribution
 * lines, cap the subject at 72 chars, and keep at most a short body.
 */
export function cleanCommitMessage(raw: string): string {
  let s = (raw ?? "").trim();
  s = s.replace(/^```[a-z]*\s*/i, "").replace(/\s*```$/, "");
  s = s.replace(/^(?:commit\s+message|message|subject)\s*:\s*/i, "");
  s = s.replace(/^["'`]|["'`]$/g, "").trim();
  const lines = s
    .split(/\r?\n/)
    .filter((l) => !/^\s*(co-authored-by|generated with|🤖)/i.test(l));
  let subject = (lines.shift() ?? "").trim().replace(/^[-*]\s*/, "").replace(/\.$/, "");
  if (subject.length > 72) subject = subject.slice(0, 69).trimEnd() + "…";
  const body = lines.join("\n").trim().slice(0, 1000);
  return body ? `${subject}\n\n${body}` : subject;
}

/**
 * Parse `git status --porcelain` lines into file paths. Must tolerate a
 * leading space having been trimmed (" M a.txt" → "M a.txt"), otherwise a
 * fixed-width slice eats the first character of the filename. Renames
 * ("R  old -> new") report the destination.
 */
export function parsePorcelain(status: string): string[] {
  const out: string[] = [];
  for (const raw of status.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (!line.trim()) continue;
    const m = line.match(/^\s*[ MADRCU?!]{1,2}\s+(.+)$/);
    const file = (m ? m[1]! : line.trim()).trim();
    const renamed = file.split(" -> ");
    out.push((renamed[renamed.length - 1] ?? file).replace(/^"|"$/g, ""));
  }
  return out;
}

/** Diff text trimmed for a prompt: head-biased, hard byte cap. */
export function diffForPrompt(diff: string, maxBytes = 12_000): string {
  if (diff.length <= maxBytes) return diff;
  return diff.slice(0, maxBytes) + `\n[diff truncated — ${diff.length - maxBytes} more chars]`;
}

// --- git plumbing ----------------------------------------------------------

function git(cwd: string, args: string[]): { ok: boolean; out: string } {
  try {
    const p = Bun.spawnSync(["git", "-C", cwd, ...args], { stdout: "pipe", stderr: "pipe" });
    const out = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim();
    return { ok: p.exitCode === 0, out };
  } catch (e) {
    return { ok: false, out: (e as Error).message };
  }
}

export function gitState(config: Config): GitState {
  const cwd = config.cwd;
  if (!git(cwd, ["rev-parse", "--git-dir"]).ok) {
    return { repo: false, branch: "", protectedBranch: false, dirty: [] };
  }
  const branch = git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"]).out || "HEAD";
  const status = git(cwd, ["status", "--porcelain"]).out;
  const dirty = parsePorcelain(status).slice(0, 50);
  return { repo: true, branch, protectedBranch: isProtectedBranch(branch), dirty };
}

/** Unified diff of everything uncommitted (staged + unstaged), for drafting. */
export function workingDiff(config: Config): string {
  const staged = git(config.cwd, ["diff", "--staged"]).out;
  const unstaged = git(config.cwd, ["diff"]).out;
  const untracked = git(config.cwd, ["ls-files", "--others", "--exclude-standard"]).out;
  let out = [staged, unstaged].filter(Boolean).join("\n");
  if (untracked) out += `\n[new files]\n${untracked}`;
  return out;
}

export function createBranch(config: Config, name: string): { ok: boolean; out: string } {
  return git(config.cwd, ["switch", "-c", name]);
}

/** Stage everything and commit. Never pushes; never signs off for the user. */
export function commitAll(config: Config, message: string): { ok: boolean; out: string } {
  const add = git(config.cwd, ["add", "-A"]);
  if (!add.ok) return add;
  const c = git(config.cwd, ["commit", "-m", message]);
  if (!c.ok) return c;
  const hash = git(config.cwd, ["rev-parse", "--short", "HEAD"]).out;
  return { ok: true, out: hash };
}
