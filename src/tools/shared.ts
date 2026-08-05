// Shared tool utilities: ignores, path handling, binary sniffing, redaction, truncation.
import { isAbsolute, relative, resolve, basename, join } from "node:path";
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

// ── Sandbox boundary (guardrail, not a VM: no symlink resolution, and bash
// containment is pattern-based — see scanDangerous in bash.ts) ──────────────

/** Directories the agent may mutate freely: workspace, data dir, session plans. */
export function sandboxRoots(config: Config): string[] {
  const roots = [config.cwd, config.dataDir];
  if (config.sessionId) roots.push(join(tmpdir(), ".ap", config.sessionId));
  return roots;
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
  if (isPrivatePath(path, ctx.config)) {
    throw new ToolError(
      `denied: ${path} is AP-private data (session transcripts, checkpoints, credentials) — never readable. Stay within ${ctx.config.cwd}.`,
    );
  }
  if (isInsideRoots(path, readRoots(ctx.config))) return;
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
  if (isPrivatePath(path, ctx.config)) {
    throw new ToolError(`denied: ${path} is AP-private data (session transcripts, checkpoints, credentials) — never writable.`);
  }
  if (isInsideRoots(path, sandboxRoots(ctx.config))) return;
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

/** Keep head + tail, elide the middle. */
export function truncateMiddle(s: string, maxBytes: number): string {
  if (Buffer.byteLength(s, "utf8") <= maxBytes) return s;
  const half = Math.floor(maxBytes / 2);
  const head = s.slice(0, half);
  const tail = s.slice(-half);
  const cut = Buffer.byteLength(s, "utf8") - maxBytes;
  return `${head}\n[... ${cut} bytes truncated ...]\n${tail}`;
}

export class ToolError extends Error {}

/** Combine an abort signal with a timeout, tolerating runtimes without
 *  AbortSignal.any (falls back to the timeout alone). */
export function anySignal(a: AbortSignal | undefined, b: AbortSignal): AbortSignal {
  if (!a) return b;
  const any = (AbortSignal as any).any;
  return typeof any === "function" ? any([a, b]) : b;
}
