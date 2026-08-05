// Subagent tool: delegates a task to a child `ap run --json --light` process.
// Children are always the light profile, so they cannot spawn further agents
// (structural recursion cap). Zero deps — we spawn our own binary.
import { existsSync } from "node:fs";
import { ensureAllowed, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

export interface SubagentInfo {
  id: number;
  task: string;
  status: "running" | "done" | "error" | "killed";
  steps: number;
  startedAt: number;
  endedAt?: number;
}

const registry: SubagentInfo[] = [];
let nextId = 1;

export function listSubagents(): SubagentInfo[] {
  return registry;
}

/** How to re-invoke ourselves: compiled exe vs `bun src/index.ts`.
 * Compiled binaries embed the entry at a VIRTUAL path — "/$bunfs/…" on
 * unix, "B:\~BUN\…" on Windows — so the reliable test is: does the entry
 * exist on disk? If not, we're compiled and the exe alone re-runs itself. */
function selfCmd(): string[] {
  const entry = process.argv[1] ?? "";
  if (!entry || entry.includes("$bunfs") || entry.includes("~BUN") || !existsSync(entry)) {
    return [process.execPath];
  }
  return [process.execPath, entry];
}

export async function agentTool(
  args: { task: string; name?: string; cwd?: string; timeout?: number },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.task !== "string" || !args.task.trim()) {
    throw new ToolError("agent requires {task}");
  }
  const cwd = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  // The child gets `cwd` as ITS sandbox root, so an unchecked cwd was a total
  // escape: agent({cwd:"C:\\"}) handed a subagent the whole drive, with the
  // parent's permission rules gone. Gate it like any other outside access.
  if (args.cwd) await ensureAllowed(cwd, ctx, "delegate a subagent outside the workspace");
  const timeoutMs = Math.min((args.timeout ?? 300) * 1000, 900_000);

  // Named profile: validated HERE so a typo fails fast with the valid list
  // instead of dying inside the child process.
  let profileArgs: string[] = [];
  if (args.name) {
    const { discoverAgents } = await import("../agents.ts");
    const defs = discoverAgents(ctx.config);
    const def = defs.find((a) => a.name === String(args.name).toLowerCase());
    if (!def) {
      throw new ToolError(
        `unknown agent profile "${args.name}" — available: ${defs.map((a) => a.name).join(", ") || "(none defined)"}`,
      );
    }
    profileArgs = ["--agent", def.name];
  }

  const info: SubagentInfo = {
    id: nextId++,
    task: `${args.name ? `[${args.name}] ` : ""}${args.task.replace(/\s+/g, " ")}`.slice(0, 100),
    status: "running",
    steps: 0,
    startedAt: Date.now(),
  };
  registry.push(info);
  ctx.subline?.(`◇ agent #${info.id} started: ${info.task.slice(0, 70)}`);

  const proc = Bun.spawn(
    [...selfCmd(), "run", "-p", args.task, "--json", "--light", "--cwd", cwd, ...profileArgs],
    { stdout: "pipe", stderr: "pipe", stdin: "ignore", windowsHide: true } as any,
  );
  // Child stderr is the only place startup-phase crashes surface — keep a tail.
  let stderrTail = "";
  const stderrDone = (async () => {
    try {
      for await (const chunk of proc.stderr as any) {
        stderrTail = (stderrTail + new TextDecoder().decode(chunk)).slice(-1000);
      }
    } catch {}
  })();

  let killed = false;
  const timer = setTimeout(() => { killed = true; proc.kill(); }, timeoutMs);
  const onAbort = () => { killed = true; proc.kill(); };
  ctx.signal.addEventListener("abort", onAbort, { once: true });

  let finalText = "";
  let errorMsg = "";
  try {
    const reader = proc.stdout.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        try {
          const e = JSON.parse(line);
          if (e.type === "tool_end") {
            info.steps++;
            ctx.subline?.(`  ↳ ${e.error ? "✗" : "✓"} ${e.name} · ${e.ms}ms`);
          } else if (e.type === "done") {
            finalText = e.text ?? "";
          } else if (e.type === "error" && !e.retryable) {
            errorMsg = e.message ?? "";
          }
        } catch {}
      }
    }
    await proc.exited;
    await stderrDone;
  } finally {
    clearTimeout(timer);
    ctx.signal.removeEventListener("abort", onAbort);
  }

  info.endedAt = Date.now();
  const secs = ((info.endedAt - info.startedAt) / 1000).toFixed(1);
  if (killed) {
    info.status = "killed";
    ctx.subline?.(`◇ agent #${info.id} killed after ${secs}s`);
    throw new ToolError(`subagent #${info.id} timed out or was aborted after ${secs}s`);
  }
  // A child that produced no final text failed, whatever the exit code says —
  // surface every clue we have (error event, stderr tail, exit code).
  if (!finalText) {
    const code = proc.exitCode;
    const clue = errorMsg || stderrTail.trim() || `exit code ${code}, no output`;
    if (errorMsg || code !== 0) {
      info.status = "error";
      ctx.subline?.(`◇ agent #${info.id} failed after ${secs}s`);
      throw new ToolError(`subagent #${info.id} failed: ${truncateMiddle(clue, 600)}`);
    }
  }
  info.status = "done";
  ctx.subline?.(`◇ agent #${info.id} done · ${info.steps} steps · ${secs}s`);
  return truncateMiddle(finalText || "(subagent produced no output)", 20_000);
}
