// Shared tool utilities: ignores, path handling, binary sniffing, redaction, truncation.
import { realpathSync } from "node:fs";
import { isAbsolute, relative, resolve, basename, dirname, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import type { Config } from "../config.ts";
import type { ToolCtx } from "./index.ts";

// Matched as path segments anywhere in a path — build-server/builds, trash, etc.
export const HARD_IGNORES = [
  "node_modules",
  ".git",
  "dist",
  "dist-wp",
  "dist-theme",
  "builds",
  "trash",
  "uploads",
  ".vercel",
  ".claude",
  "temp",
  "site-state",
];

export function allIgnores(config: Config): string[] {
  return [...HARD_IGNORES, ...config.ignore];
}

/** True if any path segment matches an ignore entry. */
export function isIgnoredPath(relOrAbs: string, ignores: string[]): boolean {
  const segs = relOrAbs.split(/[\\/]/);
  return segs.some((s) => ignores.includes(s));
}

export function resolvePath(p: string, cwd: string): string {
  if (process.platform === "win32") {
    // Models on Windows sometimes emit Git-Bash style paths (/c/Users/...).
    const m = p.match(/^\/([A-Za-z])\/(.*)$/);
    if (m) p = `${m[1]!.toUpperCase()}:/${m[2]}`;
    // Strip Windows device prefixes (\\?\C:\… and \\.\C:\…, plus their UNC
    // forms). They address the SAME files but slip past every containment
    // check, which made them a full sandbox bypass.
    p = p.replace(/^\\\\[?.]\\UNC\\/i, "\\\\").replace(/^\\\\[?.]\\(?=[A-Za-z]:[\\/])/, "");
  }
  return isAbsolute(p) ? resolve(p) : resolve(cwd, p);
}

// ── Sandbox boundary (guardrail, not a VM: bash containment is still
// pattern-based — see scanDangerous in bash.ts — but file tools resolve
// symlinks before the containment check so a workspace link cannot point
// at credentials or /etc) ───────────────────────────────────────────────────

/**
 * Directories the agent may mutate freely.
 * Deliberately NOT the whole dataDir: that let the model plant skills,
 * workflows, agents, and overwrite background/task indexes. Memory +
 * artifacts are the only dataDir subtrees the prompt asks it to write.
 * (When dataDir itself sits inside the workspace — e.g. `./.ap` — the
 * workspace root alone would still allow those writes; see
 * `isRestrictedDataDirPath`.)
 */
export function sandboxRoots(config: Config): string[] {
  const roots = [
    config.cwd,
    join(config.dataDir, "memory"),
    join(config.dataDir, "artifacts"),
  ];
  if (config.sessionId) roots.push(join(tmpdir(), ".ap", config.sessionId));
  return roots;
}

/** dataDir subtrees the model may write without a permit. */
export function dataDirWriteRoots(config: Config): string[] {
  return [join(config.dataDir, "memory"), join(config.dataDir, "artifacts")];
}

/**
 * True when `target` is under dataDir but outside memory/artifacts.
 * Hard-denied for writes even if dataDir is nested inside the workspace
 * (otherwise narrowing sandboxRoots is a no-op for `./.ap` layouts).
 */
export function isRestrictedDataDirPath(target: string, config: Config): boolean {
  if (!isInsideRoots(target, [config.dataDir])) return false;
  return !isInsideRoots(target, dataDirWriteRoots(config));
}

export function isInsideRoots(target: string, roots: string[]): boolean {
  const t = process.platform === "win32" ? target.toLowerCase() : target;
  for (const root of roots) {
    const r = process.platform === "win32" ? root.toLowerCase() : root;
    const rel = relative(r, t);
    if (rel === "" || (!rel.startsWith("..") && !isAbsolute(rel))) return true;
  }
  return false;
}

/**
 * Resolve symlinks/junctions before containment checks. For paths that do
 * not exist yet (write/create), resolve the nearest existing ancestor and
 * rejoin the remainder — otherwise `ln -s /etc workspace/x` + `read x/passwd`
 * would pass a workspace-relative check and then follow the link.
 */
export function canonicalPath(target: string): string {
  try {
    return realpathSync(target);
  } catch {
    // Walk up until an ancestor exists (or we hit the root).
    let rest = basename(target);
    let dir = dirname(target);
    for (;;) {
      try {
        return join(realpathSync(dir), rest);
      } catch {
        const parent = dirname(dir);
        if (parent === dir) return target; // filesystem root, nothing to resolve
        rest = join(basename(dir), rest);
        dir = parent;
      }
    }
    return target;
  }
}

/** Directories the agent may READ freely: workspace + skills/memory/commands + session plans. */
export function readRoots(config: Config): string[] {
  const roots = [
    config.cwd,
    join(config.dataDir, "skills"),
    join(config.dataDir, "memory"),
    join(config.dataDir, "commands"),
    join(homedir(), ".claude", "skills"),
  ];
  if (config.sessionId) roots.push(join(tmpdir(), ".ap", config.sessionId));
  return roots;
}

/** The four AP-private locations (transcripts, checkpoints, credentials, config). */
export function privatePaths(config: Config): string[] {
  return [
    join(config.dataDir, "sessions"),
    join(config.dataDir, "checkpoints"),
    join(config.dataDir, "credentials.json"),
    join(config.dataDir, "config.json"),
  ];
}

/** AP-private data: session transcripts, checkpoints, credentials, config.
 * NEVER readable by the model — permission cannot override, only sandbox:"off". */
export function isPrivatePath(target: string, config: Config): boolean {
  return isInsideRoots(target, privatePaths(config));
}

/**
 * ripgrep --glob exclusions that keep a SEARCH from walking into AP-private
 * data. `read` hard-denies those paths, but grep/glob only gated the search
 * ROOT — so a search rooted above the data dir (running ap from your home
 * directory is enough) happily printed credentials.json and past transcripts.
 *
 * Two forms per entry are required: `!rel` matches the file itself, `!rel/**`
 * matches a directory's contents. Separators must be forward slashes — a
 * backslash inside an rg glob is an escape, so Windows paths silently fail.
 */
export function privateExcludeGlobs(config: Config, searchRoot: string): string[] {
  const out: string[] = [];
  for (const p of privatePaths(config)) {
    let rel = relative(searchRoot, p);
    if (!rel || rel.startsWith("..") || isAbsolute(rel)) continue; // not under this root
    rel = rel.replace(/\\/g, "/");
    out.push("-g", `!${rel}`, "-g", `!${rel}/**`);
  }
  return out;
}

/** Gate a read: private → hard deny; outside read roots → permission. */
export async function ensureReadable(path: string, ctx: ToolCtx): Promise<void> {
  if (ctx.config.sandbox === "off") return;
  const canon = canonicalPath(path);
  // Check BOTH the lexical path and the symlink target — a link whose name
  // sits inside the workspace must not reveal private data behind it.
  if (isPrivatePath(path, ctx.config) || isPrivatePath(canon, ctx.config)) {
    throw new ToolError(
      `denied: ${path} is AP-private data (session transcripts, checkpoints, credentials) — never readable. Stay within ${ctx.config.cwd}.`,
    );
  }
  if (isInsideRoots(canon, readRoots(ctx.config)) || isInsideRoots(path, readRoots(ctx.config))) {
    // Still require the canonical target to be inside read roots when it
    // differs (symlink escape to /etc, ~/.ssh, …).
    if (canon !== path && !isInsideRoots(canon, readRoots(ctx.config))) {
      const okLink = await ctx.permit({ action: "read outside workspace (symlink)", detail: `${path} → ${canon}`, path: canon });
      if (!okLink) {
        throw new ToolError(
          `denied: ${path} resolves outside the workspace (→ ${canon}) — remove the symlink or work within ${ctx.config.cwd}`,
        );
      }
    }
    return;
  }
  const ok = await ctx.permit({ action: "read outside workspace", detail: path, path });
  if (!ok) {
    throw new ToolError(
      `denied: ${path} is outside the workspace — do not explore unrelated folders; work within ${ctx.config.cwd}` +
      ` (the user can approve interactively, or pass --allow-outside in headless mode)`,
    );
  }
}

/** Gate a mutating file action: inside the sandbox → free; outside → permission. */
export async function ensureAllowed(path: string, ctx: ToolCtx, action: string): Promise<void> {
  if (ctx.config.sandbox === "off") return;
  const canon = canonicalPath(path);
  if (isPrivatePath(path, ctx.config) || isPrivatePath(canon, ctx.config)) {
    throw new ToolError(`denied: ${path} is AP-private data (session transcripts, checkpoints, credentials) — never writable.`);
  }
  // Even when dataDir lives inside the workspace, only memory/ + artifacts/
  // are model-writable. Permits cannot override — same tier as private paths.
  if (isRestrictedDataDirPath(path, ctx.config) || isRestrictedDataDirPath(canon, ctx.config)) {
    throw new ToolError(
      `denied: ${path} is under AP's data dir — only memory/ and artifacts/ are writable there (skills, agents, workflows, logs are not).`,
    );
  }
  if (isInsideRoots(canon, sandboxRoots(ctx.config)) || isInsideRoots(path, sandboxRoots(ctx.config))) {
    if (canon !== path && !isInsideRoots(canon, sandboxRoots(ctx.config))) {
      const okLink = await ctx.permit({ action: `${action} outside workspace (symlink)`, detail: `${path} → ${canon}`, path: canon });
      if (!okLink) {
        throw new ToolError(
          `denied: ${path} resolves outside the workspace sandbox (→ ${canon}) — work within ${ctx.config.cwd}`,
        );
      }
    }
    return;
  }
  const ok = await ctx.permit({ action, detail: path, path });
  if (!ok) {
    throw new ToolError(
      `denied: ${path} is outside the workspace sandbox — work within ${ctx.config.cwd}` +
      ` (the user can approve interactively, pass --allow-outside in headless mode, or set sandbox:"off")`,
    );
  }
}

const BINARY_EXTS = new Set([
  "png", "jpg", "jpeg", "gif", "webp", "avif", "ico", "bmp",
  "ttf", "otf", "woff", "woff2", "eot",
  "zip", "gz", "tar", "rar", "7z",
  "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
  "mp3", "mp4", "wav", "mov", "avi", "webm",
  "exe", "dll", "so", "dylib", "node", "wasm",
  "db", "sqlite", "sqlite3",
]);

export function looksBinaryByExt(path: string): boolean {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return BINARY_EXTS.has(ext);
}

export function sniffBinary(bytes: Uint8Array): boolean {
  const n = Math.min(bytes.length, 8192);
  for (let i = 0; i < n; i++) if (bytes[i] === 0) return true;
  return false;
}

export function isEnvFile(path: string): boolean {
  const name = basename(path).toLowerCase();
  return name === ".env" || name.startsWith(".env.");
}

/** Common secret-bearing filenames beyond `.env` (read + grep redaction). */
export function isSecretFile(path: string): boolean {
  if (isEnvFile(path)) return true;
  const name = basename(path).toLowerCase();
  if (name === "credentials.json" || name === "credentials.yml" || name === "credentials.yaml") return true;
  if (name === "secrets.json" || name === "secrets.yml" || name === "secrets.yaml") return true;
  if (/\.(pem|p12|pfx|key)$/i.test(name)) return true;
  if (name === "id_rsa" || name === "id_ed25519" || name === "id_ecdsa" || name === "id_dsa") return true;
  return false;
}

/** Mask values in .env content: KEY=*** (model sees keys, never values). */
export function redactEnvContent(content: string): string {
  // Split on ALL line-ending flavours. Splitting on "\n" alone left a trailing
  // "\r" on every line of a CRLF file, and the value pattern below cannot
  // match it (`.` excludes \r, and `$` without /m is end-of-input) — so on
  // Windows, where CRLF is the norm, redaction silently did nothing at all.
  return content
    .split(/\r\n|\r|\n/)
    .map((line) => {
      const m = line.match(/^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=)(.*)$/);
      return m && m[2]!.trim() ? `${m[1]}***` : line;
    })
    .join("\n");
}

/**
 * Redact a single ripgrep content line when it came from a secret file.
 * Formats: `path:line:text` (match) and `path-line-text` (context).
 */
export function redactGrepLine(line: string): string {
  const m = line.match(/^(.+?)([:-])(\d+)\2(.*)$/);
  if (!m) return line;
  const file = m[1]!;
  if (!isSecretFile(file) && !isEnvFile(file)) return line;
  const body = m[4]!;
  // Prefer KEY=*** when it looks like an env assignment; otherwise mask the
  // whole payload so pem/key material never reaches the model via grep.
  const env = body.match(/^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=)(.*)$/);
  const redacted = env && env[2]!.trim()
    ? `${env[1]}***`
    : (body.trim() ? " ***" : body);
  return `${file}${m[2]}${m[3]}${m[2]}${redacted}`;
}

/**
 * Keep head + tail, elide the middle. Cuts on BYTES, not UTF-16 units: the cap
 * is a byte budget, so slicing by characters let CJK/emoji-heavy output run
 * ~3x over every documented cap (a "50KB" fetch could return ~150KB). Cut
 * points snap back to a UTF-8 lead byte so no character is split in half.
 */
export function truncateMiddle(s: string, maxBytes: number): string {
  const bytes = new TextEncoder().encode(s);
  if (bytes.length <= maxBytes) return s;
  const half = Math.max(0, Math.floor(maxBytes / 2) - 40); // leave room for the marker
  const isCont = (b: number | undefined) => b !== undefined && (b & 0xc0) === 0x80;

  let end = half;
  while (end > 0 && isCont(bytes[end])) end--; // don't split a character

  let start = bytes.length - half;
  while (start < bytes.length && isCont(bytes[start])) start++;

  const dec = new TextDecoder();
  const cut = bytes.length - maxBytes;
  return `${dec.decode(bytes.subarray(0, end))}\n[... ${cut} bytes truncated ...]\n${dec.decode(bytes.subarray(start))}`;
}

export class ToolError extends Error {}

/** Combine an abort signal with a timeout, tolerating runtimes without
 *  AbortSignal.any (falls back to the timeout alone). */
export function anySignal(a: AbortSignal | undefined, b: AbortSignal): AbortSignal {
  if (!a) return b;
  const any = (AbortSignal as any).any;
  return typeof any === "function" ? any([a, b]) : b;
}
