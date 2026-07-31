// Shared tool utilities: ignores, path handling, binary sniffing, redaction, truncation.
import { isAbsolute, resolve, basename } from "node:path";
import type { Config } from "../config.ts";

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
  return isAbsolute(p) ? resolve(p) : resolve(cwd, p);
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
  return content
    .split("\n")
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
