// Cross-process live status. Subagents, background tasks and flows live in
// the memory of ONE ap process, so a second terminal cannot see them. Each
// process therefore publishes a small snapshot to <dataDir>/live/<pid>.json,
// and `ap watch` renders every live process from those files.
//
// Rules that keep this cheap and safe:
//  - Writes are throttled and best-effort: publishing status must never slow
//    a turn or throw into it.
//  - Write-then-rename, so a reader never sees a half-written file.
//  - Readers prune snapshots whose pid is gone (same liveness check bg.ts
//    uses) and anything older than STALE_MS, so crashes self-clean.
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const STALE_MS = 60_000;
const MIN_WRITE_MS = 400;

export interface LiveAgent {
  id: number;
  label: string;
  status: string;
  steps: number;
  startedAt: number;
  background?: boolean;
  /** Task text and final output, so the viewer can show what a subagent
   *  actually did. Capped by the publisher. */
  fullTask?: string;
  result?: string;
  /** Tail of live reasoning/text while the subagent is still running. */
  live?: string;
}

/** A detached shell process started by `bash background:true` (bg.ts). These
 *  are NOT subagents — different mechanism, different lifetime — but they are
 *  the other thing "running in the background", so the viewer shows both. */
export interface LiveProc {
  pid: number;
  cmd: string;
  log: string;
  alive: boolean;
  bytes: number | null;
}

export interface LiveSnapshot {
  pid: number;
  at: number;
  cwd: string;
  session: string;
  model: string;
  /** Turn state: what the process is doing right now. */
  busy: boolean;
  ctxPct?: number;
  usd?: number;
  /** Running/finished subagents and background tasks. */
  agents: LiveAgent[];
  /** Detached shell processes (bash background:true). */
  procs?: LiveProc[];
  /** Name of the workflow currently running, if any. */
  flow?: string;
  flowStep?: string;
}

function dir(dataDir: string): string {
  return join(dataDir, "live");
}

let lastWrite = 0;

/** Publish this process's snapshot. Throttled; never throws. */
export function publishLive(dataDir: string, snap: Omit<LiveSnapshot, "pid" | "at">, force = false): void {
  const now = Date.now();
  if (!force && now - lastWrite < MIN_WRITE_MS) return;
  lastWrite = now;
  try {
    const d = dir(dataDir);
    mkdirSync(d, { recursive: true });
    const file = join(d, `${process.pid}.json`);
    const tmp = `${file}.tmp`;
    writeFileSync(tmp, JSON.stringify({ pid: process.pid, at: now, ...snap }));
    renameSync(tmp, file); // atomic swap — readers never see a partial file
  } catch {}
}

/** Remove this process's snapshot (called on exit). Never throws. */
export function clearLive(dataDir: string): void {
  try { rmSync(join(dir(dataDir), `${process.pid}.json`), { force: true }); } catch {}
}

/** Same liveness probe as bg.ts: signal 0 tests existence without signalling. */
export function pidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch (e: any) { return e?.code === "EPERM"; }
}

/**
 * Every live snapshot, newest first. Prunes files whose process is gone or
 * whose timestamp is stale — a crashed process cleans itself up on the next
 * read rather than lingering forever.
 */
export function readLive(dataDir: string): LiveSnapshot[] {
  const d = dir(dataDir);
  if (!existsSync(d)) return [];
  const out: LiveSnapshot[] = [];
  for (const f of readdirSync(d)) {
    if (!f.endsWith(".json")) continue;
    const p = join(d, f);
    try {
      const snap = JSON.parse(readFileSync(p, "utf8")) as LiveSnapshot;
      const dead = !pidAlive(snap.pid) || Date.now() - snap.at > STALE_MS;
      if (dead) { try { rmSync(p, { force: true }); } catch {} continue; }
      out.push(snap);
    } catch {
      try { rmSync(p, { force: true }); } catch {} // corrupt/partial — drop it
    }
  }
  return out.sort((a, b) => b.at - a.at);
}

/** One-line summary per process, for `ap watch` and /watch. */
export function formatSnapshot(s: LiveSnapshot, now = Date.now()): string[] {
  const age = Math.max(0, Math.round((now - s.at) / 1000));
  const head = `pid ${s.pid} · ${s.model} · ${s.busy ? "working" : "idle"}${s.ctxPct != null ? ` · ctx ${s.ctxPct}%` : ""}${s.usd ? ` · ~$${s.usd.toFixed(3)}` : ""}${age > 5 ? ` · ${age}s ago` : ""}`;
  const lines = [head, `  ${s.cwd}  ${s.session}`];
  if (s.flow) lines.push(`  ◆ flow ${s.flow}${s.flowStep ? ` · ${s.flowStep}` : ""}`);
  for (const a of s.agents.slice(-8)) {
    const secs = Math.round((now - a.startedAt) / 1000);
    lines.push(`  ◇ #${a.id} [${a.status}]${a.background ? " &" : ""} ${a.label.slice(0, 60)} · ${a.steps} steps · ${secs}s`);
  }
  return lines;
}
