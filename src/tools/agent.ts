// Subagent tool: delegates a task to a child `ap run --json --light` process.
// Children are always the light profile, so they cannot spawn further agents
// (structural recursion cap). Zero deps — we spawn our own binary.
// background:true detaches the child from the turn: it keeps running after
// the tool returns, and its result reaches the model as a note folded into
// the next user message (tasks.ts).
import { ensureAllowed, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import { runSubagent } from "./subagent.ts";
import { auditTask, finishTask, registerTask } from "../tasks.ts";
import { postChannel, safeChannelId } from "../channels.ts";
import type { ToolCtx } from "./index.ts";

export interface SubagentInfo {
  id: number;
  task: string;
  status: "running" | "done" | "error" | "killed";
  steps: number;
  startedAt: number;
  endedAt?: number;
  background?: boolean;
  /** Full task text and final output, so /watch can show what a subagent
   *  actually did instead of only that it ran. Capped — this is mirrored into
   *  the cross-process snapshot on every heartbeat. */
  fullTask?: string;
  result?: string;
  /** Tail of the subagent's live reasoning/text while it runs, so /watch can
   *  show its thinking instead of a black box. Bounded — tail only. */
  live?: string;
}

/** Keep only the last N chars, so a chatty subagent cannot grow memory. */
const LIVE_CAP = 8000;
function appendLive(info: SubagentInfo, delta: string): void {
  const next = (info.live ?? "") + delta;
  info.live = next.length > LIVE_CAP ? next.slice(next.length - LIVE_CAP) : next;
}

const registry: SubagentInfo[] = [];
let nextId = 1;

export function listSubagents(): SubagentInfo[] {
  return registry;
}

/** Validate a named profile HERE so a typo fails fast with the valid list
 *  instead of dying inside the child process. */
async function profileArgsFor(name: string | undefined, ctx: ToolCtx): Promise<string[]> {
  if (!name) return [];
  const { discoverAgents } = await import("../agents.ts");
  const defs = discoverAgents(ctx.config);
  const def = defs.find((a) => a.name === String(name).toLowerCase());
  if (!def) {
    throw new ToolError(
      defs.length
        ? `unknown agent profile "${name}" — available: ${defs.map((a) => a.name).join(", ")}. Omit "name" to run a plain subagent.`
        : `no agent profiles are defined, so "name" cannot be used. Drop the "name" argument and put the role in the task itself (e.g. task: "search the web for X and summarize"). Profiles are files: .ap/agents/<name>.md`,
    );
  }
  // Project/claude-repo profiles only after trust — otherwise a cloned repo
  // plants .ap/agents/evil.md and the model selects it.
  if (def.source !== "global" && ctx.config.workspaceTrusted !== true) {
    throw new ToolError(
      `agent profile "${name}" is from this workspace, which is untrusted — ask the user to trust it before using project agents`,
    );
  }
  return ["--agent", def.name];
}

export async function agentTool(
  args: { task: string; name?: string; cwd?: string; timeout?: number; background?: boolean; channel?: string },
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
  const extraArgs = await profileArgsFor(args.name, ctx);
  const channel = args.channel ? safeChannelId(String(args.channel)) : null;
  if (args.channel && !channel) throw new ToolError(`invalid channel id "${args.channel}" — use [\\w-]{1,64}`);
  const from = args.name ? `agent:${args.name}` : "agent";
  let taskText = args.task;
  if (channel) {
    taskText =
      `[channel:${channel}] Post brief progress or findings by mentioning them clearly in your final answer — the parent reads your result onto this channel.\n\n` +
      args.task;
  }
  const label = `${args.name ? `[${args.name}] ` : ""}${args.task.replace(/\s+/g, " ")}`.slice(0, 100);

  const publish = (text: string) => {
    if (channel) postChannel(ctx.config.dataDir, channel, from, text.slice(0, 2000));
  };

  if (args.background) {
    // Detached: the task gets its OWN abort controller — the turn's ctrl+c
    // must not kill it — and reports back via the next-turn note queue.
    const t = registerTask(label, timeoutMs);
    const info: SubagentInfo = {
      id: nextId++, task: label, status: "running", steps: 0,
      startedAt: t.startedAt, background: true, fullTask: args.task.slice(0, 2000),
    };
    registry.push(info);
    auditTask(ctx.config, t, "start");
    ctx.subline?.(`◇ task #${t.id} started in background: ${label.slice(0, 70)}`);
    void runSubagent({
      task: taskText, cwd, timeoutMs, extraArgs,
      signal: t.ctrl.signal,
      onStep: () => { t.steps++; info.steps++; },
      onOutput: (_kind, delta) => appendLive(info, delta),
    }).then((r) => {
      const failed = r.killed || (!r.text && (r.errorMsg || r.exitCode !== 0));
      const status = r.killed ? "killed" : failed ? "error" : "done";
      const result = r.killed
        ? "killed (timeout or /tasks kill)"
        : failed
          ? truncateMiddle(r.errorMsg || r.stderrTail.trim() || `exit code ${r.exitCode}, no output`, 600)
          : truncateMiddle(r.text || "(subagent produced no output)", 20_000);
      if (!failed && !r.killed) publish(result);
      finishTask(t, status, result);
      info.status = status;
      info.result = result.slice(0, 4000);
      info.endedAt = t.endedAt;
      auditTask(ctx.config, t, "end");
    }).catch((e) => {
      finishTask(t, "error", String((e as Error).message ?? e));
      info.status = "error";
      info.result = String((e as Error).message ?? e).slice(0, 4000);
      info.endedAt = Date.now();
      auditTask(ctx.config, t, "end");
    });
    return `started background task #${t.id}${channel ? ` on channel ${channel}` : ""} — the result will arrive with the next turn (/tasks to inspect)`;
  }

  const info: SubagentInfo = {
    id: nextId++, task: label, status: "running", steps: 0, startedAt: Date.now(),
    fullTask: args.task.slice(0, 2000),
  };
  registry.push(info);
  ctx.subline?.(`◇ agent #${info.id} started: ${info.task.slice(0, 70)}`);

  const r = await runSubagent({
    task: taskText, cwd, timeoutMs, extraArgs,
    signal: ctx.signal,
    onStep: (line) => { info.steps++; ctx.subline?.(line); },
    onOutput: (_kind, delta) => appendLive(info, delta),
  });

  info.endedAt = Date.now();
  const secs = ((info.endedAt - info.startedAt) / 1000).toFixed(1);
  if (r.killed) {
    info.status = "killed";
    info.result = `killed or timed out after ${secs}s`;
    ctx.subline?.(`◇ agent #${info.id} killed after ${secs}s`);
    throw new ToolError(`subagent #${info.id} timed out or was aborted after ${secs}s`);
  }
  // A child that produced no final text failed, whatever the exit code says —
  // surface every clue we have (error event, stderr tail, exit code).
  if (!r.text) {
    const clue = r.errorMsg || r.stderrTail.trim() || `exit code ${r.exitCode}, no output`;
    if (r.errorMsg || r.exitCode !== 0) {
      info.status = "error";
      info.result = truncateMiddle(clue, 2000);
      ctx.subline?.(`◇ agent #${info.id} failed after ${secs}s`);
      throw new ToolError(`subagent #${info.id} failed: ${truncateMiddle(clue, 600)}`);
    }
  }
  info.status = "done";
  info.result = truncateMiddle(r.text || "(subagent produced no output)", 4000);
  publish(info.result);
  ctx.subline?.(`◇ agent #${info.id} done · ${info.steps} steps · ${secs}s`);
  return truncateMiddle(r.text || "(subagent produced no output)", 20_000);
}
