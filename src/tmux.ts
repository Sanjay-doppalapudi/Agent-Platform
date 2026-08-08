// Optional tmux adapter (unix). Native Windows has no tmux — callers get a
// clear message and should use /ps + /worktree instead. Detected via
// Bun.which("tmux"); never a hard dependency.
import { join } from "node:path";
import type { Config } from "./config.ts";
import { slugifyBranch } from "./git.ts";

export function tmuxAvailable(): boolean {
  return !!Bun.which("tmux");
}

export function tmuxMissingHint(): string {
  return process.platform === "win32"
    ? "tmux is not available on native Windows — use /worktree + bash background:true + /ps, or run under WSL"
    : "tmux not found on PATH — install tmux, or use /worktree + /ps";
}

function run(args: string[], cwd?: string): { ok: boolean; out: string } {
  try {
    const p = Bun.spawnSync(["tmux", ...args], { cwd, stdout: "pipe", stderr: "pipe" });
    const out = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim();
    return { ok: p.exitCode === 0, out };
  } catch (e) {
    return { ok: false, out: (e as Error).message };
  }
}

/** Sanitized tmux session name: ap-<slug>, capped. */
export function tmuxSessionName(hint: string): string {
  const slug = slugifyBranch(hint, "").replace(/^ap\//, "") || "work";
  return `ap-${slug}`.slice(0, 50);
}

/**
 * Detach a headless `ap run` in a new tmux session. Returns session name.
 * Uses --light for children by default (same trust tier as subagents).
 */
export function tmuxSpawn(
  config: Config,
  task: string,
  opts?: { cwd?: string; light?: boolean; name?: string },
): { ok: boolean; session: string; out: string } {
  if (!tmuxAvailable()) return { ok: false, session: "", out: tmuxMissingHint() };
  const session = opts?.name || tmuxSessionName(task);
  const cwd = opts?.cwd || config.cwd;
  // Resolve the same binary the user invoked when possible.
  const bin = process.execPath.includes("bun")
    ? `"${process.execPath}" "${join(import.meta.dir, "index.ts")}"`
    : `"${process.execPath}"`;
  const light = opts?.light === false ? "" : " --light";
  // Escape for a single-quoted sh -c payload: close, escaped quote, reopen.
  const esc = (s: string) => s.replace(/'/g, `'\\''`);
  const cmd = `${bin} run --json${light} --cwd '${esc(cwd)}' -p '${esc(task)}'`;
  // Kill any leftover same-name session so spawn is idempotent for the slug.
  run(["kill-session", "-t", session]);
  const r = run(["new-session", "-d", "-s", session, "-c", cwd, cmd]);
  if (!r.ok) return { ok: false, session, out: r.out || "tmux new-session failed" };
  return { ok: true, session, out: `tmux session ${session} started · attach: tmux attach -t ${session}` };
}

/** Bootstrap a 3-pane layout: ap | shell | logs placeholder. */
export function tmuxLayout(config: Config, sessionName = "ap"): { ok: boolean; out: string } {
  if (!tmuxAvailable()) return { ok: false, out: tmuxMissingHint() };
  const bin = process.execPath.includes("bun")
    ? `"${process.execPath}" "${join(import.meta.dir, "index.ts")}"`
    : `"${process.execPath}"`;
  run(["kill-session", "-t", sessionName]);
  // Pane 0: ap REPL; pane 1: shell; pane 2: note about /ps
  let r = run(["new-session", "-d", "-s", sessionName, "-c", config.cwd, `${bin} --cwd '${config.cwd.replace(/'/g, `'\\''`)}'`]);
  if (!r.ok) return { ok: false, out: r.out };
  run(["split-window", "-h", "-t", sessionName, "-c", config.cwd]);
  run(["split-window", "-v", "-t", `${sessionName}:0.1`, "-c", config.cwd]);
  run(["select-pane", "-t", `${sessionName}:0.0`]);
  return {
    ok: true,
    out: `tmux layout ready · attach: tmux attach -t ${sessionName}\n  pane 0: ap · pane 1: shell · pane 2: spare (pipe logs here)`,
  };
}

export function tmuxList(): { ok: boolean; out: string } {
  if (!tmuxAvailable()) return { ok: false, out: tmuxMissingHint() };
  const r = run(["list-sessions"]);
  if (!r.ok) return { ok: true, out: "(no tmux sessions)" };
  return { ok: true, out: r.out || "(no tmux sessions)" };
}

export function tmuxCapture(session: string, lines = 80): { ok: boolean; out: string } {
  if (!tmuxAvailable()) return { ok: false, out: tmuxMissingHint() };
  const r = run(["capture-pane", "-pt", session, "-S", `-${Math.max(1, Math.min(lines, 500))}`]);
  return r.ok ? r : { ok: false, out: r.out || `no session ${session}` };
}
