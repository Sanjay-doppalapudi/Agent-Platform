// Shadow-repo checkpoints: a git repo living under <dataDir>/checkpoints/<session>
// whose work-tree is the workspace. Auto-commits after mutating turns give
// unlimited /undo, /diff, /restore without touching the user's real git —
// and they work in non-git folders too. Pure `git` CLI, zero deps.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Config } from "./config.ts";
import { HARD_IGNORES } from "./tools/shared.ts";

const GIT_ID = ["-c", "user.email=ap@local", "-c", "user.name=AP"];

function git(gitDir: string, workTree: string, args: string[], quiet = true): { ok: boolean; out: string } {
  const p = Bun.spawnSync(
    ["git", "--git-dir", gitDir, "--work-tree", workTree, ...GIT_ID, ...args],
    { cwd: workTree, stdout: "pipe", stderr: "pipe" }, // cwd matters: clean/add resolve paths from here
  );
  const out = (p.stdout?.toString() ?? "") + (quiet ? "" : p.stderr?.toString() ?? "");
  return { ok: p.exitCode === 0, out: out.trim() };
}

export class Checkpoints {
  private gitDir: string;
  private ready = false;

  constructor(private config: Config, private sessionId: string) {
    this.gitDir = join(config.dataDir, "checkpoints", sessionId);
  }

  available(): boolean {
    return this.config.checkpoints !== "off" && !this.config.light && !!Bun.which("git");
  }

  private ensure(): boolean {
    if (this.ready) return true;
    if (!this.available()) return false;
    if (!existsSync(join(this.gitDir, "HEAD"))) {
      mkdirSync(this.gitDir, { recursive: true });
      const init = Bun.spawnSync(["git", "init", "--quiet", "--bare", this.gitDir], { stdout: "ignore", stderr: "ignore" });
      if (init.exitCode !== 0) return false;
      // Exclude noise dirs AND any nested .git so the real repo isn't swallowed.
      const excludes = [".git/", ...HARD_IGNORES.map((d) => `${d}/`), ...this.config.ignore.map((d) => `${d}/`)];
      mkdirSync(join(this.gitDir, "info"), { recursive: true });
      writeFileSync(join(this.gitDir, "info", "exclude"), excludes.join("\n") + "\n");
      git(this.gitDir, this.config.cwd, ["config", "core.autocrlf", "false"]);
      // `git clean` (and friends) refuse to run when core.bare=true — the
      // shadow repo is bare-initialized but always used with a work-tree.
      git(this.gitDir, this.config.cwd, ["config", "core.bare", "false"]);
      git(this.gitDir, this.config.cwd, ["config", "core.worktree", this.config.cwd]);
    }
    this.ready = true;
    return true;
  }

  /** Commit the current workspace state. Returns short hash or null. */
  commit(label: string): string | null {
    if (!this.ensure()) return null;
    const w = this.config.cwd;
    if (!git(this.gitDir, w, ["add", "-A", "."]).ok) return null;
    const staged = git(this.gitDir, w, ["diff", "--cached", "--quiet"]);
    if (staged.ok) return null; // nothing changed
    const msg = label.replace(/\s+/g, " ").slice(0, 72) || "checkpoint";
    if (!git(this.gitDir, w, ["commit", "--quiet", "-m", msg]).ok) return null;
    return git(this.gitDir, w, ["rev-parse", "--short", "HEAD"]).out || null;
  }

  /** Newest-first list of {hash, label}. */
  list(limit = 15): { hash: string; label: string }[] {
    if (!this.ensure()) return [];
    const r = git(this.gitDir, this.config.cwd, ["log", "--format=%h%x09%s", `-${limit}`]);
    if (!r.ok || !r.out) return [];
    return r.out.split("\n").map((l) => {
      const tab = l.indexOf("\t");
      return { hash: l.slice(0, tab === -1 ? undefined : tab), label: tab === -1 ? "" : l.slice(tab + 1) };
    });
  }

  /** Restore the workspace to a checkpoint (files tracked then; extras cleaned). */
  restore(hash: string): string | null {
    if (!this.ensure()) return null;
    const w = this.config.cwd;
    // read-tree makes the INDEX match the target (checkout -- . would leave
    // later-added files tracked, so clean couldn't remove them).
    if (!git(this.gitDir, w, ["read-tree", hash]).ok) return null;
    git(this.gitDir, w, ["checkout-index", "-a", "-f"]);
    git(this.gitDir, w, ["clean", "-fdq"]); // excludes (node_modules, .git, …) respected
    return this.commit(`restore ${hash}`) ?? hash;
  }

  /** Short hash of the latest checkpoint, or null before the first commit. */
  head(): string | null {
    if (!this.ensure()) return null;
    const r = git(this.gitDir, this.config.cwd, ["rev-parse", "--short", "HEAD"]);
    return r.ok && r.out ? r.out : null;
  }

  private static EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

  /** Per-file +/- line counts from a checkpoint to the current workspace
   * (committed + pending merged). from=null → since the first checkpoint. */
  filesChanged(from: string | null): { file: string; add: string; del: string }[] {
    if (!this.ensure()) return [];
    const w = this.config.cwd;
    const acc = new Map<string, { add: number; del: number; bin: boolean }>();
    const ingest = (args: string[]) => {
      const r = git(this.gitDir, w, ["diff", "--numstat", ...args]);
      if (!r.ok || !r.out) return;
      for (const line of r.out.split("\n")) {
        const [a, d, ...rest] = line.split("\t");
        const file = rest.join("\t");
        if (!file) continue;
        const e = acc.get(file) ?? { add: 0, del: 0, bin: false };
        if (a === "-" || d === "-") e.bin = true;
        else { e.add += Number(a) || 0; e.del += Number(d) || 0; }
        acc.set(file, e);
      }
    };
    ingest([from ?? Checkpoints.EMPTY_TREE, "HEAD"]);
    ingest(["HEAD"]); // pending, uncommitted
    return [...acc.entries()].map(([file, e]) => ({
      file,
      add: e.bin ? "-" : String(e.add),
      del: e.bin ? "-" : String(e.del),
    }));
  }

  /** Diff between checkpoints (n back → HEAD) plus uncommitted changes. */
  diff(back = 1, maxBytes = 30_000): string {
    if (!this.ensure()) return "(checkpoints unavailable)";
    const w = this.config.cwd;
    const committed = git(this.gitDir, w, ["diff", "--stat", "--patch", `HEAD~${back}`, "HEAD"], false);
    const pending = git(this.gitDir, w, ["diff", "--stat", "--patch", "HEAD"], false);
    let out = "";
    if (committed.out) out += committed.out;
    if (pending.out) out += (out ? "\n\n── uncommitted since last checkpoint ──\n" : "") + pending.out;
    if (!out) return "(no changes)";
    return out.length > maxBytes ? out.slice(0, maxBytes) + "\n[diff truncated]" : out;
  }
}
