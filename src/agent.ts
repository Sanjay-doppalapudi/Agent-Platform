// The agent loop: stream a completion, dispatch tool calls (reads in
// parallel, mutations serialized), append results, repeat until the model
// stops calling tools.
import type { Config, ResolvedProvider } from "./config.ts";
import { ProviderError, streamChat, type Msg } from "./provider.ts";
import { buildSystemPrompt } from "./prompt.ts";
import type { Session } from "./session.ts";
import { autoDenyPermit, execTool, getTool, isParallelSafe, permissionFor, resolveToolName, toolSchemasFor, type PermitFn } from "./tools/index.ts";
import type { Usage } from "./stream.ts";

export type AgentEvent =
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "warn"; message: string }
  | { type: "subline"; text: string }
  | { type: "tool_start"; id: string; name: string; args: unknown }
  | { type: "tool_end"; id: string; name: string; output: string; ms: number; error?: boolean }
  | { type: "turn_end"; usage?: Usage }
  | { type: "done"; sessionId: string; text: string }
  | { type: "error"; message: string; retryable: boolean };

export type Emit = (e: AgentEvent) => void;

const KEEP_RECENT_TOOL_MSGS = 12;
const MAX_HOOK_CONTINUATIONS = 2;

/** Run a config hook command; returns {ok, output}. 30s cap, never throws.
 * TRUST MODEL (see SECURITY.md): the hook COMMAND string comes exclusively
 * from the user's own config files (ap.config.json / <dataDir>/config.json)
 * — never from model output or network input. Tool arguments reach the hook
 * via environment variables (AP_TOOL/AP_ARGS), never interpolated into the
 * command line, so hostile arguments cannot inject into the shell. */
function runHook(cmd: string, cwd: string, tool: string, argsJson: string): { ok: boolean; output: string } {
  try {
    const p = Bun.spawnSync(
      process.platform === "win32" ? ["cmd", "/c", cmd] : ["sh", "-c", cmd],
      {
        cwd,
        env: { ...process.env, AP_TOOL: tool, AP_ARGS: argsJson.slice(0, 8000) },
        timeout: 30_000,
        stdout: "pipe",
        stderr: "pipe",
      } as any,
    );
    const output = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim().slice(0, 8000);
    return { ok: p.exitCode === 0, output };
  } catch (e) {
    return { ok: false, output: `hook failed to run: ${(e as Error).message}` };
  }
}

/** Fire-and-forget lifecycle hook: config `hooks.onDone` / `hooks.onError` is
 * either a shell command (payload in AP_EVENT / AP_PAYLOAD env) or an http(s)
 * URL that gets a JSON POST. Never blocks or fails the turn. Same trust model
 * as runHook: the command/URL comes only from the user's own config, and the
 * payload travels via env vars / request body — never shell interpolation. */
const pendingLifecycle = new Set<Promise<unknown>>();

function fireLifecycle(config: Config, name: "onDone" | "onError", payload: Record<string, unknown>) {
  const spec = config.light ? undefined : config.hooks?.[name];
  if (!spec) return;
  const body = JSON.stringify({ event: name, ...payload });
  let p: Promise<unknown> | null = null;
  try {
    if (/^https?:\/\//i.test(spec)) {
      p = fetch(spec, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        signal: AbortSignal.timeout(10_000),
      }).catch(() => {});
    } else {
      const proc = Bun.spawn(
        process.platform === "win32" ? ["cmd", "/c", spec] : ["sh", "-c", spec],
        {
          cwd: config.cwd,
          env: { ...process.env, AP_EVENT: name, AP_PAYLOAD: body.slice(0, 8000) },
          stdin: "ignore",
          stdout: "ignore",
          stderr: "ignore",
          windowsHide: true,
        } as any,
      );
      p = proc.exited.catch(() => {});
    }
  } catch {}
  if (p) {
    pendingLifecycle.add(p);
    p.finally(() => pendingLifecycle.delete(p!));
  }
}

/** Drain outstanding lifecycle hooks (capped). run/loop call this before
 * process.exit — on Windows, children of an exiting parent are killed. */
export async function lifecycleSettled(maxWaitMs = 10_000): Promise<void> {
  if (!pendingLifecycle.size) return;
  await Promise.race([
    Promise.allSettled([...pendingLifecycle]),
    new Promise((r) => setTimeout(r, maxWaitMs)),
  ]);
}

export interface RunOptions {
  extra?: Record<string, unknown>; // passthrough body fields (e.g. response_format)
  systemOverride?: string;
  permit?: PermitFn; // sandbox permission callback (default: deny)
}

/** Run one user turn to completion. Returns the final assistant text. */
export async function runTurn(
  config: Config,
  provider: ResolvedProvider,
  session: Session,
  userText: string,
  emit: Emit,
  signal: AbortSignal,
  opts: RunOptions = {},
): Promise<string> {
  const system = opts.systemOverride ?? buildSystemPrompt(config);
  session.append({ role: "user", content: userText });

  let finalText = "";
  let midStreamRetried = false; // one mid-stream retry per turn
  let hookContinuations = 0;
  const HOOK_PRE: Record<string, string> = { bash: "preBash", write: "preWrite", edit: "preEdit" };
  for (let iter = 0; iter < config.maxIterations; iter++) {
    if (signal.aborted) break;
    const msgs: Msg[] = [{ role: "system", content: system }, ...session.history];
    trimContext(msgs, config.contextBudgetChars);

    let res;
    try {
      res = await streamChat(
        provider,
        msgs,
        toolSchemasFor(config),
        (delta) => emit({ type: "text", delta }),
        signal,
        opts.extra,
        (delta) => emit({ type: "reasoning", delta }),
        config.streamIdleSeconds * 1000,
      );
    } catch (e) {
      const pe = e as ProviderError;
      // Mid-stream failures are safe to retry whole: tools only run after
      // full assembly and history is appended only on success.
      if (pe.retryable && pe.midStream && !midStreamRetried && !signal.aborted) {
        midStreamRetried = true;
        emit({ type: "error", message: `${pe.message} — retrying the request`, retryable: true });
        // Partial text has reached every consumer (pipe, editor, SSE) and cannot
        // be retracted, so mark the boundary in-band on the SAME channel —
        // otherwise the retry's answer reads as a continuation of the truncated
        // one ("Let me check the confLet me check the config file…").
        emit({ type: "text", delta: "\n\n[stream interrupted — the partial answer above is discarded; retrying]\n\n" });
        await new Promise((r) => setTimeout(r, 1000));
        iter--;
        continue;
      }
      emit({ type: "error", message: pe.message, retryable: pe.retryable ?? false });
      fireLifecycle(config, "onError", { sessionId: session.id, cwd: config.cwd, message: pe.message });
      throw e;
    }

    session.append({
      role: "assistant",
      content: res.text || null,
      ...(res.toolCalls.length ? { tool_calls: res.toolCalls } : {}),
    });
    emit({ type: "turn_end", usage: res.usage });
    finalText = res.text;

    if (res.toolCalls.length === 0) {
      emit({ type: "done", sessionId: session.id, text: finalText });
      fireLifecycle(config, "onDone", { sessionId: session.id, cwd: config.cwd, text: finalText.slice(0, 4000) });
      return finalText;
    }

    // Dispatch: read-only calls concurrently, mutating calls serially after,
    // results appended in the model's original order. Names are alias-resolved
    // BEFORE classification/blocking so e.g. "search" counts as read-only.
    const ctx = {
      cwd: config.cwd,
      signal,
      config,
      permit: opts.permit ?? autoDenyPermit,
      warn: (message: string) => emit({ type: "warn", message }),
      subline: (text: string) => emit({ type: "subline", text }),
    };
    const results = new Array<{ output: string; error: boolean; ms: number }>(res.toolCalls.length);
    const policy = config.parallelPolicy;

    const runOne = async (i: number) => {
      const tc = res.toolCalls[i]!;
      const canonical = resolveToolName(tc.function.name);
      let args: unknown = {};
      try { args = JSON.parse(tc.function.arguments || "{}"); } catch {}
      emit({ type: "tool_start", id: tc.id, name: canonical, args });
      const start = performance.now();
      let r: { output: string; error: boolean };
      const preHook = !config.light && config.hooks?.[HOOK_PRE[canonical] ?? ""];
      if (config.mode === "plan" && !getTool(canonical)?.readOnly) {
        // Plan mode is structurally read-only; block hallucinated mutations too.
        r = { output: "blocked: plan mode is read-only (switch to code mode to apply changes)", error: true };
      } else {
        // Per-tool permission rules first (explicit config wins), then the
        // coarse permissions mode. "allow" skips only the ask gate — the path
        // sandbox and bashGuard inside the tools still apply.
        let allowed = true;
        let denial = "denied by user";
        const rule = permissionFor(config, canonical, tc.function.arguments);
        if (rule === "deny") {
          allowed = false;
          denial = `blocked by permission config ("${canonical}": deny) — this tool/command is not permitted in this project`;
        } else if (rule === "ask" || (rule === null && config.permissions === "prompt" && getTool(canonical) && !getTool(canonical)!.readOnly)) {
          allowed = await ctx.permit({
            action: `run ${canonical}`,
            detail: `${canonical} ${tc.function.arguments.slice(0, 120)}`,
          });
        }
        if (!allowed) {
          r = { output: denial, error: true };
        } else if (preHook) {
          const h = runHook(preHook, config.cwd, canonical, tc.function.arguments);
          r = h.ok
            ? await execTool(tc.function.name, tc.function.arguments, ctx)
            : { output: `blocked by ${HOOK_PRE[canonical]} hook:\n${h.output || "(no output)"}`, error: true };
        } else {
          r = await execTool(tc.function.name, tc.function.arguments, ctx);
        }
      }
      const ms = Math.round(performance.now() - start);
      results[i] = { ...r, ms };
      emit({ type: "tool_end", id: tc.id, name: canonical, output: r.output, ms, error: r.error });
    };

    if (policy === "none") {
      for (let i = 0; i < res.toolCalls.length; i++) await runOne(i);
    } else if (policy === "all") {
      await Promise.all(res.toolCalls.map((_, i) => runOne(i)));
    } else {
      const readIdx: number[] = [];
      const mutIdx: number[] = [];
      res.toolCalls.forEach((tc, i) => {
        (isParallelSafe(tc.function.name) ? readIdx : mutIdx).push(i);
      });
      await Promise.all(readIdx.map(runOne));
      for (const i of mutIdx) await runOne(i);
    }

    for (let i = 0; i < res.toolCalls.length; i++) {
      session.append({
        role: "tool",
        tool_call_id: res.toolCalls[i]!.id,
        content: results[i]!.output,
      });
    }

    // afterEdit hook: if this iteration mutated files, run diagnostics; a
    // nonzero exit feeds the output back so the model fixes it next call.
    const afterEdit = !config.light && config.hooks?.afterEdit;
    if (afterEdit && hookContinuations < MAX_HOOK_CONTINUATIONS) {
      const mutated = res.toolCalls.some((tc, i) => {
        const c = resolveToolName(tc.function.name);
        return (c === "write" || c === "edit") && !results[i]!.error;
      });
      if (mutated) {
        const h = runHook(afterEdit, config.cwd, "afterEdit", "{}");
        if (!h.ok && h.output) {
          hookContinuations++;
          emit({ type: "warn", message: `afterEdit hook reported problems — asking the model to fix them` });
          session.append({ role: "user", content: `[afterEdit hook output — fix these issues]\n${h.output}` });
        }
      }
    }
  }

  if (signal.aborted) {
    // Keep history consistent: a dangling assistant tool_calls msg without
    // tool results breaks the next request — patch in cancellation results.
    patchDanglingToolCalls(session);
    emit({ type: "done", sessionId: session.id, text: finalText });
    // Abort is a terminal path too. onError (not onDone) — a cancel landing
    // mid-stream already fires onError, and onDone means "the chat finished".
    fireLifecycle(config, "onError", { sessionId: session.id, cwd: config.cwd, message: "aborted", reason: "aborted" });
    return finalText;
  }

  session.append({ role: "user", content: "[max iterations reached — stop and summarize]" });
  emit({ type: "error", message: `max iterations (${config.maxIterations}) reached`, retryable: false });
  emit({ type: "done", sessionId: session.id, text: finalText });
  fireLifecycle(config, "onDone", { sessionId: session.id, cwd: config.cwd, text: finalText.slice(0, 4000), reason: "max_iterations" });
  return finalText;
}

function patchDanglingToolCalls(session: Session) {
  const last = session.history[session.history.length - 1];
  if (last?.role === "assistant" && last.tool_calls?.length) {
    for (const tc of last.tool_calls) {
      session.append({ role: "tool", tool_call_id: tc.id, content: "[cancelled by user]" });
    }
  }
}

/**
 * Char-budget context trim: elide oldest tool results (keeping the most
 * recent KEEP_RECENT_TOOL_MSGS) until under budget. System/assistant text
 * untouched — preserves the cached prefix up to the first elision.
 */
export function trimContext(msgs: Msg[], budgetChars: number) {
  let size = 0;
  for (const m of msgs) size += JSON.stringify(m).length;
  if (size <= budgetChars) return;

  const toolIdxs = msgs.reduce<number[]>((acc, m, i) => {
    if (m.role === "tool") acc.push(i);
    return acc;
  }, []);
  const trimmable = toolIdxs.slice(0, Math.max(0, toolIdxs.length - KEEP_RECENT_TOOL_MSGS));

  for (const i of trimmable) {
    if (size <= budgetChars) break;
    const m = msgs[i] as Extract<Msg, { role: "tool" }>;
    if (m.content.startsWith("[elided")) continue;
    const saved = m.content.length;
    msgs[i] = { ...m, content: `[elided tool result, ${saved} chars]` };
    size -= saved - (msgs[i] as any).content.length;
  }
}
