// The agent loop: stream a completion, dispatch tool calls (reads in
// parallel, mutations serialized), append results, repeat until the model
// stops calling tools.
import type { Config, ResolvedProvider } from "./config.ts";
import { ProviderError, streamChat, type Msg } from "./provider.ts";
import { buildSystemPrompt } from "./prompt.ts";
import type { Session } from "./session.ts";
import { execTool, getTool, toolSchemasFor } from "./tools/index.ts";
import type { Usage } from "./stream.ts";

export type AgentEvent =
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "tool_start"; id: string; name: string; args: unknown }
  | { type: "tool_end"; id: string; name: string; output: string; ms: number; error?: boolean }
  | { type: "turn_end"; usage?: Usage }
  | { type: "done"; sessionId: string; text: string }
  | { type: "error"; message: string; retryable: boolean };

export type Emit = (e: AgentEvent) => void;

const KEEP_RECENT_TOOL_MSGS = 12;

export interface RunOptions {
  extra?: Record<string, unknown>; // passthrough body fields (e.g. response_format)
  systemOverride?: string;
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
  for (let iter = 0; iter < config.maxIterations; iter++) {
    if (signal.aborted) break;
    const msgs: Msg[] = [{ role: "system", content: system }, ...session.history];
    trimContext(msgs, config.contextBudgetChars);

    let res;
    try {
      res = await streamChat(
        provider,
        msgs,
        toolSchemasFor(config.mode),
        (delta) => emit({ type: "text", delta }),
        signal,
        opts.extra,
        (delta) => emit({ type: "reasoning", delta }),
      );
    } catch (e) {
      const pe = e as ProviderError;
      emit({ type: "error", message: pe.message, retryable: pe.retryable ?? false });
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
      return finalText;
    }

    // Dispatch: read-only calls concurrently, mutating calls serially after,
    // results appended in the model's original order.
    const ctx = { cwd: config.cwd, signal, config };
    const results = new Array<{ output: string; error: boolean; ms: number }>(res.toolCalls.length);
    const policy = config.parallelPolicy;

    const runOne = async (i: number) => {
      const tc = res.toolCalls[i]!;
      let args: unknown = {};
      try { args = JSON.parse(tc.function.arguments || "{}"); } catch {}
      emit({ type: "tool_start", id: tc.id, name: tc.function.name, args });
      const start = performance.now();
      // Plan mode is structurally read-only; block hallucinated mutations too.
      const r =
        config.mode === "plan" && !getTool(tc.function.name)?.readOnly
          ? { output: "blocked: plan mode is read-only (switch to code mode to apply changes)", error: true }
          : await execTool(tc.function.name, tc.function.arguments, ctx);
      const ms = Math.round(performance.now() - start);
      results[i] = { ...r, ms };
      emit({ type: "tool_end", id: tc.id, name: tc.function.name, output: r.output, ms, error: r.error });
    };

    if (policy === "none") {
      for (let i = 0; i < res.toolCalls.length; i++) await runOne(i);
    } else if (policy === "all") {
      await Promise.all(res.toolCalls.map((_, i) => runOne(i)));
    } else {
      const readIdx: number[] = [];
      const mutIdx: number[] = [];
      res.toolCalls.forEach((tc, i) => {
        (getTool(tc.function.name)?.readOnly ? readIdx : mutIdx).push(i);
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
  }

  if (signal.aborted) {
    // Keep history consistent: a dangling assistant tool_calls msg without
    // tool results breaks the next request — patch in cancellation results.
    patchDanglingToolCalls(session);
    emit({ type: "done", sessionId: session.id, text: finalText });
    return finalText;
  }

  session.append({ role: "user", content: "[max iterations reached — stop and summarize]" });
  emit({ type: "error", message: `max iterations (${config.maxIterations}) reached`, retryable: false });
  emit({ type: "done", sessionId: session.id, text: finalText });
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
