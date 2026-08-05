// Shell tool — SECURITY MODEL (full policy in SECURITY.md):
// Commands are authored by the model within a session the local user started
// and supervises; AP is a development tool, not a privilege boundary. Layered
// guardrails, all best-effort by design (a guardrail, not a VM):
//   1. scanDangerous — destructive patterns are BLOCKED outright (never
//      prompted) and logged for provider feedback
//   2. scanCmdPaths — path tokens outside the readable roots require an
//      interactive user permit; AP-private data (credentials, transcripts,
//      checkpoints) is hard-denied and cannot be permitted
//   3. timeouts with process-tree kill, output caps, ctrl+c abort,
//      background logs confined to the user's data dir with 7-day pruning
// Run genuinely untrusted code in a container/VM, not behind these checks.
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { isInsideRoots, isPrivatePath, readRoots, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

// Dangerous-command patterns: auto-BLOCKED (never prompted), warned, and
// logged to <dataDir>/blocked-commands.jsonl so the user can report the model.
// Best-effort by design — obfuscated commands can evade a pattern scan.
const DANGEROUS: [RegExp, string][] = [
  [/\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\S*\s+("|')?(\/|[A-Za-z]:[\\/]|~\/?(\s|$)|\$HOME)/i, "recursive force delete of an absolute path"],
  [/\b(del|rmdir|rd)\b[^&|;]*\/s/i, "recursive Windows delete"],
  [/\bformat\s+[a-z]:/i, "disk format"],
  [/\breg(\.exe)?\s+(add|delete)\b/i, "registry modification"],
  [/remove-item\b[^&|;]*-recurse[^&|;]*([A-Za-z]:[\\/]|\\\\)/i, "recursive Remove-Item on an absolute path"],
  [/\bmkfs\b/, "filesystem format"],
  [/\bdd\s+[^&|;]*of=\/dev\//, "raw disk write"],
  // Command position only (start / after ;|&& / sudo) — mentioning the word
  // in an echo or commit message is not an attempt to run it.
  [/(^|[;&|]\s*|\bsudo\s+)shutdown\b/i, "system shutdown/restart"],
  [/\btaskkill\b[^&|;]*\/f[^&|;]*\/im/i, "force-kill processes by name"],
  [/:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;/, "fork bomb"],
  [/\b(curl|wget|irm|iwr)\b[^&|;]*\|\s*(bash|sh|iex|powershell)/i, "piping a download straight into a shell"],
];

export function scanDangerous(cmd: string): string | null {
  for (const [re, why] of DANGEROUS) if (re.test(cmd)) return why;
  return null;
}

// Absolute-ish path tokens inside a shell command (win drive, git-bash /c/,
// unix homes, ~, $HOME, %USERPROFILE%). Best-effort — this is the same
// guardrail-not-VM stance as scanDangerous.
const PATH_TOKEN_RE = /(?:[A-Za-z]:[\\/]|\/(?:[a-z])\/|\/home\/|\/Users\/|\/etc\/|\/var\/|~[\\/]|\$HOME\b|%USERPROFILE%|\$env:USERPROFILE)[^\s"'`;|&<>()*]*/g;

/** Paths a command references outside the readable roots. {priv} = AP-private. */
export function scanCmdPaths(
  cmd: string,
  ctx: ToolCtx,
): { outside: string[]; priv: string[] } {
  const roots = readRoots(ctx.config);
  const outside = new Set<string>();
  const priv = new Set<string>();
  for (const m of cmd.match(PATH_TOKEN_RE) ?? []) {
    let p = m.replace(/^~(?=[\\/])/, homedir()).replace(/^(\$HOME|%USERPROFILE%|\$env:USERPROFILE)/, homedir());
    p = resolvePath(p.replace(/[.,:]+$/, ""), ctx.cwd);
    if (isPrivatePath(p, ctx.config)) priv.add(p);
    else if (!isInsideRoots(p, roots)) outside.add(p);
  }
  // Relative escapes: ../ chains or a bare `cd ..` step out of the workspace.
  if (/(?:^|[\s"'=(])\.\.[\\/]|\bcd\s+\.\.(?![\w.])/.test(cmd)) outside.add("(relative path escaping the workspace via ..)");
  return { outside: [...outside].slice(0, 5), priv: [...priv].slice(0, 5) };
}

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
  if (typeof args.cmd !== "string" || !args.cmd.trim()) {
    throw new ToolError("bash requires {cmd}");
  }
  if (ctx.config.bashGuard !== "off") {
    const danger = scanDangerous(args.cmd);
    if (danger) {
      const logPath = join(ctx.config.dataDir, "blocked-commands.jsonl");
      try {
        mkdirSync(ctx.config.dataDir, { recursive: true });
        appendFileSync(logPath, JSON.stringify({
          at: new Date().toISOString(),
          sessionId: ctx.config.sessionId ?? null,
          cwd: ctx.cwd,
          cmd: args.cmd,
          reason: danger,
        }) + "\n");
      } catch {}
      ctx.warn?.(`dangerous command blocked (${danger}) — logged to ${logPath}; share that log with your model provider`);
      throw new ToolError(`dangerous command blocked: ${danger}. Do not retry it or attempt workarounds.`);
    }
  }
  if (ctx.config.sandbox !== "off") {
    const { outside, priv } = scanCmdPaths(args.cmd + (args.cwd ? ` ${args.cwd}` : ""), ctx);
    if (priv.length) {
      throw new ToolError(
        `denied: command references AP-private data (${priv.join(", ")}) — session transcripts, checkpoints and credentials are never accessible. Stay within ${ctx.cwd}.`,
      );
    }
    if (outside.length) {
      const ok = await ctx.permit({
        action: "bash outside workspace",
        detail: `${args.cmd.slice(0, 120)} → ${outside.join(", ")}`,
        path: outside[0],
      });
      if (!ok) {
        throw new ToolError(
          `denied: command references paths outside the workspace (${outside.join(", ")}) — do not explore unrelated folders; work within ${ctx.cwd}` +
          ` (the user can approve interactively, or pass --allow-outside in headless mode)`,
        );
      }
    }
  }
  const cwd = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  const prefix = shellPrefix(ctx.config.shell);

  if (args.background) {
    const logDir = join(ctx.config.dataDir, "logs");
    mkdirSync(logDir, { recursive: true });
    // Retention: background logs can contain sensitive command output — prune
    // anything older than 7 days whenever a new background process starts.
    try {
      const cutoff = Date.now() - 7 * 24 * 3600 * 1000;
      for (const f of readdirSync(logDir)) {
        if (f.startsWith("bg-") && f.endsWith(".log") && statSync(join(logDir, f)).mtimeMs < cutoff) {
          rmSync(join(logDir, f), { force: true });
        }
      }
    } catch {}
    const logFile = join(logDir, `bg-${Date.now()}.log`);
    // Redirect INSIDE the shell command, not via an inherited fd: on Windows
    // a raw fd passed through stdio is silently dropped for detached children
    // (verified with both node:child_process and Bun.spawn — logs stayed 0
    // bytes), so the shell itself must open the file.
    const redirected =
      prefix[0]!.includes("powershell")
        ? `& { ${args.cmd} } *> "${logFile}"`
        : prefix[0]!.includes("cmd")
          ? `( ${args.cmd} ) > "${logFile}" 2>&1`
          : `{ ${args.cmd} ; } > "${logFile.replace(/\\/g, "/")}" 2>&1`;
    const child = spawn(prefix[0]!, [...prefix.slice(1), redirected], {
      cwd,
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    child.unref();
    // Register it so /ps (and `ap ps`) can find, tail, and kill it later —
    // detached children outlive this process and would otherwise be orphans.
    if (!ctx.config.light && child.pid) {
      const { recordBackground } = await import("../bg.ts");
      recordBackground(ctx.config, {
        pid: child.pid,
        cmd: args.cmd.replace(/\s+/g, " ").slice(0, 200),
        cwd,
        log: logFile,
        sessionId: ctx.config.sessionId ?? null,
      });
    }
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
