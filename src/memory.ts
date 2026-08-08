// Repo-keyed memory directories. Worktrees of the same git repo share one
// memory pool (via --git-common-dir); unrelated projects under the same
// dataDir do not. Soft-reads top-level legacy <dataDir>/memory/*.md when the
// keyed dir is empty so existing notes are not orphaned.
import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";

const MEMORY_CHAR_CAP = 2000;

/** Stable directory segment for a workspace. */
export function repoMemoryKey(cwd: string): string {
  const common = gitCommonDir(cwd);
  const basis = common ?? resolve(cwd);
  const norm = process.platform === "win32" ? basis.toLowerCase() : basis;
  const prefix = common ? "git" : "cwd";
  return `${prefix}-${createHash("sha256").update(norm).digest("hex").slice(0, 16)}`;
}

/** Absolute memory directory for this cwd under dataDir. */
export function repoMemoryDir(dataDir: string, cwd: string): string {
  return join(dataDir, "memory", repoMemoryKey(cwd));
}

/** Ensure the keyed dir exists (writes go here). */
export function ensureRepoMemoryDir(dataDir: string, cwd: string): string {
  const dir = repoMemoryDir(dataDir, cwd);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function gitCommonDir(cwd: string): string | null {
  try {
    const r = Bun.spawnSync(["git", "-C", cwd, "rev-parse", "--git-common-dir"], {
      stdout: "pipe",
      stderr: "pipe",
    });
    if (r.exitCode !== 0) return null;
    const raw = Buffer.from(r.stdout).toString("utf8").trim();
    if (!raw) return null;
    const abs = isAbsolute(raw) ? raw : resolve(cwd, raw);
    return resolve(abs);
  } catch {
    return null;
  }
}

/** Concatenated saved memories, capped. Prefer keyed dir; fall back to flat. */
export function readMemories(memDir: string, legacyDir?: string): string {
  let out = readMdDir(memDir);
  if (!out && legacyDir && legacyDir !== memDir) {
    out = readMdDir(legacyDir, /* topLevelOnly */ true);
  }
  return out;
}

function readMdDir(dir: string, topLevelOnly = false): string {
  let out = "";
  try {
    for (const f of readdirSync(dir).sort()) {
      if (!f.endsWith(".md")) continue;
      const p = join(dir, f);
      if (topLevelOnly) {
        try {
          if (!statSync(p).isFile()) continue;
        } catch {
          continue;
        }
      }
      out += readFileSync(p, "utf8").trim() + "\n---\n";
      if (out.length > MEMORY_CHAR_CAP) {
        return out.slice(0, MEMORY_CHAR_CAP) + "\n[more memories truncated]";
      }
    }
  } catch {}
  return out;
}

export function memoryDirHasNotes(dir: string): boolean {
  try {
    return readdirSync(dir).some((f) => f.endsWith(".md"));
  } catch {
    return false;
  }
}

export function legacyMemoryDir(dataDir: string): string {
  return join(dataDir, "memory");
}

export { MEMORY_CHAR_CAP };
