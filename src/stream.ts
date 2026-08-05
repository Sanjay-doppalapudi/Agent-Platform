// SSE parser + chat-completions delta assembler.
// Text deltas are emitted immediately; tool_call fragments are assembled by
// index (first fragment carries id+name, later fragments append arguments).

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface Usage {
  prompt: number;
  completion: number;
  cached?: number;
}

export interface AssembledResponse {
  text: string;
  toolCalls: ToolCall[];
  finishReason: string | null;
  usage?: Usage;
}

interface PartialCall {
  id: string;
  name: string;
  args: string;
}

/**
 * Streaming filter for reasoning models that inline `<think>…</think>` in
 * content (MiniMax, DeepSeek-R1 style). Reasoning is separated from the
 * answer: it is surfaced via the `think` channel (rendered dim in the UI)
 * but never included in the returned text — sending it back as history
 * wastes tokens and providers advise against it.
 */
class ThinkFilter {
  private state: "start" | "think" | "pass" = "start";
  private carry = "";

  push(delta: string): { out: string; think: string } {
    let out = "";
    let think = "";
    let s = this.carry + delta;
    this.carry = "";
    for (;;) {
      if (this.state === "start") {
        const probe = s.trimStart();
        if (probe.length < 7 && "<think>".startsWith(probe)) { this.carry = s; return { out, think }; }
        if (probe.startsWith("<think>")) {
          this.state = "think";
          s = probe.slice(7);
          continue;
        }
        this.state = "pass";
        continue;
      }
      if (this.state === "think") {
        const end = s.indexOf("</think>");
        if (end === -1) {
          // Tag may split across deltas: hold back the last 8 chars, stream the rest.
          think += s.length > 8 ? s.slice(0, -8) : "";
          this.carry = s.slice(-8);
          return { out, think };
        }
        think += s.slice(0, end);
        this.state = "pass";
        s = s.slice(end + 8).replace(/^\n+/, "");
        continue;
      }
      out += s;
      return { out, think };
    }
  }

  flush(): string {
    const rest = this.state === "pass" || this.state === "start" ? this.carry : "";
    this.carry = "";
    return rest;
  }
}

/** Thrown when the provider stops sending bytes mid-stream (idle timeout). */
export class StreamStallError extends Error {
  constructor(idleMs: number) {
    super(`stream stalled: no data from provider for ${Math.round(idleMs / 1000)}s`);
  }
}

/** Thrown when a 200 response carried no parsable SSE event at all — a proxy
 *  that ignored stream:true, an upstream that closed at byte 0, or an error
 *  frame we cannot read. Retryable: never a legitimate empty answer. */
export class StreamEmptyError extends Error {
  constructor() {
    super("provider returned no stream events (empty or non-SSE response body)");
  }
}

/**
 * Consume a chat-completions SSE body, emitting text deltas via onText and
 * returning the fully assembled response. Tolerates providers that send a
 * whole tool call in one delta, omit finish_reason, or add unknown fields.
 * idleTimeoutMs: abort if no bytes (including keep-alive comments) arrive
 * for that long — a hung provider becomes a bounded, retryable failure.
 */
export async function consumeSSE(
  body: ReadableStream<Uint8Array>,
  onText: (delta: string) => void,
  signal?: AbortSignal,
  onReasoning?: (delta: string) => void,
  idleTimeoutMs = 0,
): Promise<AssembledResponse> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const filter = new ThinkFilter();
  let buf = "";
  let text = "";
  let finishReason: string | null = null;
  let usage: Usage | undefined;
  const calls = new Map<number, PartialCall>();
  // Slot bookkeeping for providers that omit `index` on tool_calls (e.g. ones
  // that deliver a complete `message` on the last chunk). Without this every
  // index-free call landed in bucket 0 and they concatenated into one corrupt
  // call named e.g. "readgrep".
  const slotById = new Map<string, number>();
  let nextSlot = 0;
  let curSlot = -1; // most recently touched slot: continuation fragments belong here
  let sawEvent = false;

  const abort = () => reader.cancel().catch(() => {});
  signal?.addEventListener("abort", abort, { once: true });

  const readChunk = (): ReturnType<typeof reader.read> => {
    if (!idleTimeoutMs) return reader.read();
    let timer: ReturnType<typeof setTimeout>;
    const stall = new Promise<never>((_, rej) => {
      timer = setTimeout(() => {
        reader.cancel().catch(() => {});
        rej(new StreamStallError(idleTimeoutMs));
      }, idleTimeoutMs);
    });
    return Promise.race([reader.read(), stall]).finally(() => clearTimeout(timer!)) as ReturnType<typeof reader.read>;
  };

  // One SSE line. Hoisted so the trailing buffer can be flushed after the read
  // loop — a provider whose final `data:` line has no newline used to lose that
  // entire event silently.
  const handleLine = (rawLine: string): void => {
    const line = rawLine.replace(/\r$/, "");
    if (!line.startsWith("data:")) return;
    const data = line.slice(5).trim();
    if (!data) return;
    if (data === "[DONE]") { sawEvent = true; return; }

    let json: any;
    try {
      json = JSON.parse(data);
    } catch {
      return; // partial/garbage line — skip
    }
    sawEvent = true;

    if (json.usage) {
      usage = {
        prompt: json.usage.prompt_tokens ?? 0,
        completion: json.usage.completion_tokens ?? 0,
        cached: json.usage.prompt_tokens_details?.cached_tokens,
      };
    }
    const choice = json.choices?.[0];
    if (!choice) return;
    if (choice.finish_reason) finishReason = choice.finish_reason;

    const delta = choice.delta ?? choice.message; // some providers send message on last chunk
    if (!delta) return;
    // Provider-specific reasoning channels: surfaced to the UI, never
    // stored in text (so never re-sent as history).
    const reasoning = delta.reasoning_content ?? delta.reasoning;
    if (typeof reasoning === "string" && reasoning.length > 0) {
      onReasoning?.(reasoning);
    }
    if (typeof delta.content === "string" && delta.content.length > 0) {
      const { out, think } = filter.push(delta.content);
      if (think) onReasoning?.(think);
      if (out) {
        text += out;
        onText(out);
      }
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        // Three-way slotting: an explicit index wins; an unseen id starts a
        // new slot; a fragment with neither continues the current slot
        // (that is what plain argument-continuation deltas look like).
        let idx: number;
        if (typeof tc.index === "number") idx = tc.index;
        else if (tc.id) idx = slotById.get(tc.id) ?? nextSlot++;
        else idx = curSlot >= 0 ? curSlot : nextSlot++;
        if (tc.id) slotById.set(tc.id, idx);
        curSlot = idx;
        if (idx >= nextSlot) nextSlot = idx + 1; // keep synthetic slots clear of provider indices
        let partial = calls.get(idx);
        if (!partial) {
          partial = { id: "", name: "", args: "" };
          calls.set(idx, partial);
        }
        if (tc.id) partial.id = tc.id;
        if (tc.function?.name) partial.name += tc.function.name;
        if (tc.function?.arguments) partial.args += tc.function.arguments;
      }
    }
  };

  try {
    for (;;) {
      const { done, value } = await readChunk();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        handleLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    }
    buf += decoder.decode(); // flush any partial multi-byte character
    if (buf.trim()) handleLine(buf); // final line without a trailing newline
  } finally {
    signal?.removeEventListener("abort", abort);
    reader.releaseLock?.();
  }

  // A 200 whose body carried no parsable SSE event is a failed request, not an
  // empty answer. Returning success here wrote {"role":"assistant",content:null}
  // into the session and exited 0, so nothing upstream ever noticed.
  if (!sawEvent) {
    throw new StreamEmptyError();
  }

  const tail = filter.flush();
  if (tail) {
    text += tail;
    onText(tail);
  }

  const toolCalls: ToolCall[] = [...calls.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([i, c]) => ({
      id: c.id || `call_${i}`,
      type: "function" as const,
      function: { name: c.name, arguments: c.args || "{}" },
    }))
    .filter((c) => c.function.name);

  return { text, toolCalls, finishReason, usage };
}
