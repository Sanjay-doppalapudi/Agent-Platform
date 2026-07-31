import { spawn } from "node:child_process";
import { existsSync, mkdirSync, openSync } from "node:fs";
import { join } from "node:path";
import { resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_TIMEOUT_MS = 600_000;
const MAX_OUTPUT_BYTES = 30_000;

let cachedShell: string[] | null = null;

/** Resolve the shell command prefix once per process. */
export function shellPrefix(pref: "auto" | "bash" | "powershell" | "cmd"): string[] {
  if (cachedShell && pref === "auto") return cachedShell;
  const gitBashCandidates = [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files\\Git\\usr\\bin\\bash.exe",
    "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
  ];
  const resolveBash = (): string | null => {
    for (const c of gitBashCandidates) if (existsSync(c)) return c;
    const w = Bun.which("bash");
    // System32 bash is the WSL shim — not usable as a project shell.
    if (w && !/system32/i.test(w)) return w;
    return null;
  };
  let out: string[];
  switch (pref) {
    case "bash": {
      const b = resolveBash();
      if (!b) throw new ToolError("shell=bash configured but Git Bash not found");
      out = [b, "-c"];
      break;
    }
    case "powershell":
      out = ["powershell", "-NoProfile", "-Command"];
      break;
    case "cmd":
      out = ["cmd", "/c"];
      break;
    default: {
      const b = process.platform === "win32" ? resolveBash() : Bun.which("bash");
      out = b ? [b, "-c"] : process.platform === "win32" ? ["powershell", "-NoProfile", "-Command"] : ["sh", "-c"];
    }
  }
  if (pref === "auto") cachedShell = out;
  return out;
}

function killTree(pid: number) {
  if (process.platform === "win32") {
    try { Bun.spawnSync(["taskkill", "/pid", String(pid), "/t", "/f"]); } catch {}
  } else {
    try { process.kill(-pid, "SIGKILL"); } catch { try { process.kill(pid, "SIGKILL"); } catch {} }
  }
}

export async function bashTool(
  args: { cmd: string; cwd?: string; timeout?: number; background?: boolean },
  ctx: ToolCtx,
): Promise<string> {
  const cwd = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  const prefix = shellPrefix(ctx.config.shell);

  if (args.background) {
    const logDir = join(ctx.config.dataDir, "logs");
    mkdirSync(logDir, { recursive: true });
    const logFile = join(logDir, `bg-${Date.now()}.log`);
    const fd = openSync(logFile, "a");
    const child = spawn(prefix[0]!, [...prefix.slice(1), args.cmd], {
      cwd,
      detached: true,
      stdio: ["ignore", fd, fd],
      windowsHide: true,
    });
    child.unref();
    return `started background process pid=${child.pid} log=${logFile}`;
  }

  const timeoutMs = Math.min((args.timeout ?? 120) * 1000 || DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS);
  const proc = Bun.spawn([...prefix, args.cmd], {
    cwd,
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
    windowsHide: true,
  } as any);

  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; killTree(proc.pid); }, timeoutMs);
  const onAbort = () => killTree(proc.pid);
  ctx.signal.addEventListener("abort", onAbort, { once: true });

  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    let out = stdout;
    if (stderr.trim()) out += (out ? "\n" : "") + stderr;
    out = truncateMiddle(out.trim(), MAX_OUTPUT_BYTES);
    if (timedOut) return `[timed out after ${timeoutMs / 1000}s — process killed]\n${out}`;
    if (exitCode !== 0) return `[exit code ${exitCode}]\n${out}`;
    return out || "(no output)";
  } finally {
    clearTimeout(timer);
    ctx.signal.removeEventListener("abort", onAbort);
  }
}
