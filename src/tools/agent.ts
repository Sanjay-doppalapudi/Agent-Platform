// Subagent tool: delegates a task to a child `ap run --json --light` process.
// Children are always the light profile, so they cannot spawn further agents
// (structural recursion cap). Zero deps — we spawn our own binary.
import { resolvePath, truncateMiddle, ToolError } from "./shared.ts";
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

/** How to re-invoke ourselves: compiled exe vs `bun src/index.ts`. */
function selfCmd(): string[] {
  const entry = process.argv[1] ?? "";
  if (!entry || entry.includes("$bunfs")) return [process.execPath];
  return [process.execPath, entry];
}

export async function agentTool(
  args: { task: string; cwd?: string; timeout?: number },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.task !== "string" || !args.task.trim()) {
    throw new ToolError("agent requires {task}");
  }
  const cwd = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  const timeoutMs = Math.min((args.timeout ?? 300) * 1000, 900_000);

  const info: SubagentInfo = {
    id: nextId++,
    task: args.task.replace(/\s+/g, " ").slice(0, 100),
    status: "running",
    steps: 0,
    startedAt: Date.now(),
  };
  registry.push(info);
  ctx.subline?.(`◇ agent #${info.id} started: ${info.task.slice(0, 70)}`);

  const proc = Bun.spawn(
    [...selfCmd(), "run", "-p", args.task, "--json", "--light", "--cwd", cwd],
    { stdout: "pipe", stderr: "ignore", stdin: "ignore", windowsHide: true } as any,
  );

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
  if (!finalText && errorMsg) {
    info.status = "error";
    ctx.subline?.(`◇ agent #${info.id} failed after ${secs}s`);
    throw new ToolError(`subagent #${info.id} failed: ${errorMsg}`);
  }
  info.status = "done";
  ctx.subline?.(`◇ agent #${info.id} done · ${info.steps} steps · ${secs}s`);
  return truncateMiddle(finalText || "(subagent produced no output)", 20_000);
}
