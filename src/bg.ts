// Background-process registry: `bash background:true` spawns detached children
// that outlive the AP process (dev servers, watchers, long builds). Without a
// record of them they become orphans nobody can find. This is an append-only
// JSONL index at <dataDir>/background.jsonl — same no-database stance as
// sessions. Powers `/ps` in the REPL and `ap ps` from the CLI.
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Config } from "./config.ts";

export interface BgEntry {
  pid: number;
  cmd: string;
  cwd: string;
  log: string;
  at: string;
  sessionId?: string | null;
}

export interface BgStatus extends BgEntry {
  alive: boolean;
  /** Log size in bytes, or null when the file is gone. */
  bytes: number | null;
}

const INDEX = "background.jsonl";
const KEEP_DEAD_MS = 7 * 24 * 3600 * 1000; // match the bg-log retention
const MAX_ENTRIES = 200;

function indexPath(config: Config): string {
  return join(config.dataDir, INDEX);
}

/** Record a newly spawned background process (best-effort, never throws). */
export function recordBackground(config: Config, e: Omit<BgEntry, "at">): void {
  try {
    mkdirSync(config.dataDir, { recursive: true });
    appendFileSync(indexPath(config), JSON.stringify({ ...e, at: new Date().toISOString() }) + "\n");
  } catch {}
}

/** Is the process still running? (signal 0 probes without signalling.) */
export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    // EPERM = exists but owned by someone else; ESRCH = gone.
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
}

function readEntries(config: Config): BgEntry[] {
  const p = indexPath(config);
  if (!existsSync(p)) return [];
  const out: BgEntry[] = [];
  try {
    for (const line of readFileSync(p, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const e = JSON.parse(line) as BgEntry;
        if (typeof e.pid === "number" && e.log) out.push(e);
      } catch {} // torn line — skip, same as sessions
    }
  } catch {}
  return out;
}

/**
 * Registry contents, newest first, annotated with liveness. Dead entries older
 * than the log-retention window are dropped (and the index rewritten) so it
 * never grows unbounded.
 */
export function listBackground(config: Config): BgStatus[] {
  const entries = readEntries(config);
  const now = Date.now();
  const kept: BgEntry[] = [];
  const rows: BgStatus[] = [];
  for (const e of entries) {
    const alive = isAlive(e.pid);
    let bytes: number | null = null;
    try { bytes = statSync(e.log).size; } catch {}
    const age = now - Date.parse(e.at || "");
    const stale = !alive && (Number.isNaN(age) || age > KEEP_DEAD_MS);
    if (stale) continue;
    kept.push(e);
    rows.push({ ...e, alive, bytes });
  }
  if (kept.length !== entries.length || kept.length > MAX_ENTRIES) {
    const trimmed = kept.slice(-MAX_ENTRIES);
    try {
      writeFileSync(indexPath(config), trimmed.map((e) => JSON.stringify(e)).join("\n") + (trimmed.length ? "\n" : ""));
    } catch {}
  }
  return rows.reverse();
}

/** Human log size. Never rounds a small/empty log up to "1KB" — an empty log
 *  is a real signal (buffered output, or the process died before flushing). */
export function formatBytes(bytes: number | null): string {
  if (bytes == null) return "no log";
  if (bytes === 0) return "empty";
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)}KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

/** Last `lines` lines of a process's log (capped bytes read from the tail). */
export function tailLog(path: string, lines = 40, maxBytes = 64_000): string {
  try {
    const size = statSync(path).size;
    const fd = readFileSync(path); // logs are pruned + small; simple read is fine
    const text = size > maxBytes ? fd.subarray(size - maxBytes).toString("utf8") : fd.toString("utf8");
    const all = text.split(/\r?\n/);
    return all.slice(Math.max(0, all.length - lines)).join("\n").trim() || "(no output yet)";
  } catch (e) {
    return `(log unavailable: ${(e as Error).message})`;
  }
}

/** Kill a recorded background process (and its tree on Windows). */
export function killBackground(config: Config, pid: number): { ok: boolean; message: string } {
  const known = readEntries(config).some((e) => e.pid === pid);
  if (!known) return { ok: false, message: `pid ${pid} is not an AP background process` };
  if (!isAlive(pid)) return { ok: false, message: `pid ${pid} is already stopped` };
  try {
    if (process.platform === "win32") {
      Bun.spawnSync(["taskkill", "/pid", String(pid), "/t", "/f"], { stdout: "ignore", stderr: "ignore" });
    } else {
      try { process.kill(-pid, "SIGTERM"); } catch { process.kill(pid, "SIGTERM"); }
    }
    return { ok: true, message: `killed pid ${pid}` };
  } catch (e) {
    return { ok: false, message: `kill failed: ${(e as Error).message}` };
  }
}
