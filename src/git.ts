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

/**
 * Diff against a base ref (branch/tag/commit). Prefers three-dot (merge-base)
 * when the ref resolves; falls back to two-dot. Pure plumbing — callers cap.
 */
export function branchDiff(config: Config, base: string): { ok: boolean; out: string } {
  const ref = base.trim();
  if (!ref || /[\s;|&<>]/.test(ref)) return { ok: false, out: "invalid base ref" };
  if (!git(config.cwd, ["rev-parse", "--git-dir"]).ok) {
    return { ok: false, out: "not a git repository" };
  }
  // Resolve the ref first so typos don't produce an empty "ok" diff.
  if (!git(config.cwd, ["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]).ok &&
      !git(config.cwd, ["rev-parse", "--verify", "--quiet", ref]).ok) {
    return { ok: false, out: `unknown ref: ${ref}` };
  }
  const three = git(config.cwd, ["diff", `${ref}...HEAD`]);
  if (three.ok) return three;
  return git(config.cwd, ["diff", `${ref}..HEAD`]);
}

/** True when `/diff` args mean "git working tree" rather than checkpoint N. */
export function isWorkingDiffArg(arg: string | undefined): boolean {
  if (!arg) return false;
  const a = arg.toLowerCase();
  return a === "git" || a === "working" || a === "--git" || a === "-g";
}

/** True when `/diff` args look like a branch/ref (not a checkpoint back-N). */
export function isBranchDiffArg(arg: string | undefined): boolean {
  if (!arg) return false;
  if (isWorkingDiffArg(arg)) return false;
  if (/^\d+$/.test(arg)) return false; // checkpoint back-count
  // Reject shell metacharacters; allow branch-ish names and SHAs.
  return /^[A-Za-z0-9._/@~^+-]+$/.test(arg);
}

export function createBranch(config: Config, name: string): { ok: boolean; out: string } {
  return git(config.cwd, ["switch", "-c", name]);
}

/** Per-process: only auto-branch once per cwd. */
const autoBranched = new Set<string>();

/**
 * If `git.autoBranch` is on and we're on a protected branch, create
 * `ap/<slug>` before the first mutation. Idempotent per cwd for the process.
 * Returns the new branch name, or null when skipped/failed.
 */
export function maybeAutoBranch(
  config: Config,
  hint: string,
  notify?: (msg: string) => void,
): string | null {
  if (config.light || !config.git?.autoBranch) return null;
  const key = config.cwd;
  if (autoBranched.has(key)) return null;
  // Latch before any git I/O so parallel mutating tools don't race two creates.
  autoBranched.add(key);
  const st = gitState(config);
  if (!st.repo || !st.protectedBranch) return null;
  // Already on an ap/ branch somehow (rebased rename etc.) — leave it.
  if (st.branch.startsWith("ap/")) return null;
  const name = slugifyBranch(hint || config.sessionId || "work");
  // If the branch already exists, switch to it instead of failing create.
  let r = createBranch(config, name);
  if (!r.ok) {
    const sw = git(config.cwd, ["switch", name]);
    if (sw.ok) r = sw;
  }
  if (r.ok) {
    notify?.(`auto-branch → ${name} (git.autoBranch; protected "${st.branch}" left alone)`);
    return name;
  }
  notify?.(`auto-branch failed (${name}): ${r.out.slice(0, 200)}`);
  return null;
}

/** Reset auto-branch latch (tests). */
export function resetAutoBranchLatch(): void {
  autoBranched.clear();
}

/** Prefer upstream / origin/HEAD / main / master as PR base. */
export function defaultPrBase(config: Config): string {
  const upstream = git(config.cwd, ["rev-parse", "--abbrev-ref", "@{upstream}"]);
  if (upstream.ok && upstream.out.includes("/")) {
    // origin/main → main
    return upstream.out.split("/").slice(1).join("/") || "main";
  }
  for (const cand of ["main", "master", "develop"]) {
    if (git(config.cwd, ["rev-parse", "--verify", "--quiet", `origin/${cand}`]).ok ||
        git(config.cwd, ["rev-parse", "--verify", "--quiet", cand]).ok) {
      return cand;
    }
  }
  return "main";
}

/** Build title + body for `gh pr create` from the branch diff. Pure text. */
export function prPromptMaterial(config: Config, base?: string): { base: string; head: string; diff: string; commits: string } {
  const st = gitState(config);
  const b = (base?.trim() || defaultPrBase(config));
  const diff = diffForPrompt(branchDiff(config, b).out || "", 10_000);
  const commits = git(config.cwd, ["log", "--oneline", `${b}..HEAD`]).out.slice(0, 4000);
  return { base: b, head: st.branch, diff, commits };
}

/** Run `gh pr create`. Never force-pushes. */
export function createPullRequest(
  config: Config,
  opts: { title: string; body: string; base?: string; draft?: boolean },
): { ok: boolean; out: string } {
  if (!Bun.which("gh")) return { ok: false, out: "gh not found on PATH — install GitHub CLI" };
  const st = gitState(config);
  if (!st.repo) return { ok: false, out: "not a git repository" };
  if (st.protectedBranch) {
    return { ok: false, out: `on protected branch "${st.branch}" — switch to an ap/… or feature branch first` };
  }
  const base = opts.base?.trim() || defaultPrBase(config);
  const args = [
    "pr", "create",
    "--title", opts.title.slice(0, 200),
    "--body", opts.body.slice(0, 50_000),
    "--base", base,
    "--head", st.branch,
  ];
  if (opts.draft) args.push("--draft");
  try {
    const p = Bun.spawnSync(["gh", ...args], {
      cwd: config.cwd,
      stdout: "pipe",
      stderr: "pipe",
      env: process.env,
    });
    const out = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim();
    return { ok: p.exitCode === 0, out };
  } catch (e) {
    return { ok: false, out: (e as Error).message };
  }
}

/** Stage everything and commit. Never pushes; never signs off for the user. */
export function commitAll(
  config: Config,
  message: string,
  opts?: { stagedOnly?: boolean; sign?: boolean },
): { ok: boolean; out: string } {
  if (!opts?.stagedOnly) {
    const add = git(config.cwd, ["add", "-A"]);
    if (!add.ok) return add;
  } else {
    const st = git(config.cwd, ["diff", "--cached", "--quiet"]);
    // exit 1 means there is a staged diff; exit 0 means empty
    if (st.ok) return { ok: false, out: "nothing staged — stage files first, or omit --staged" };
  }
  const args = ["commit", "-m", message];
  if (opts?.sign) args.splice(1, 0, "-S");
  const c = git(config.cwd, args);
  if (!c.ok) return c;
  const hash = git(config.cwd, ["rev-parse", "--short", "HEAD"]).out;
  return { ok: true, out: hash };
}
