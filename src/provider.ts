// Fetch-based OpenAI-compatible streaming client. No SDK.
import type { ResolvedProvider } from "./config.ts";
import { consumeSSE, StreamEmptyError, StreamStallError, type AssembledResponse, type ToolCall } from "./stream.ts";

export type ContentPart = {
  type: "text";
  text: string;
  cache_control?: { type: "ephemeral" };
};

export type Msg =
  | { role: "system" | "user"; content: string | ContentPart[] }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

export interface ToolSchema {
  type: "function";
  function: { name: string; description: string; parameters: object };
}

export class ProviderError extends Error {
  constructor(
    message: string,
    public status: number,
    public retryable: boolean,
    public midStream = false, // failed after deltas started (safe to retry the whole request — tools only run after full assembly)
  ) {
    super(message);
  }
}

// Providers that reject stream_options get it dropped for the rest of the process.
let streamOptionsOk = true;

const MAX_ATTEMPTS = 3;

/**
 * Stream one chat completion. Retries (backoff + jitter) on 429/5xx/network
 * errors that occur before any delta is received. Emits text deltas via onText.
 */
export async function streamChat(
  provider: ResolvedProvider,
  messages: Msg[],
  tools: ToolSchema[],
  onText: (delta: string) => void,
  signal?: AbortSignal,
  extra?: Record<string, unknown>,
  onReasoning?: (delta: string) => void,
  idleTimeoutMs = 0,
): Promise<AssembledResponse> {
  const url = `${provider.baseUrl}/chat/completions`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${provider.apiKey}`,
    ...provider.headers,
  };

  let lastErr: Error | null = null;
  let retryAfterMs: number | undefined;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    if (signal?.aborted) throw new ProviderError("aborted", 0, false);
    if (attempt > 0) {
      const delay = 500 * 2 ** (attempt - 1) * (1 + Math.random() * 0.3);
      await sleep(retryAfterMs ?? delay, signal);
    }
    retryAfterMs = undefined;

    const body: Record<string, unknown> = {
      model: provider.model,
      messages: withCacheControl(provider, messages),
      stream: true,
      ...(streamOptionsOk ? { stream_options: { include_usage: true } } : {}),
      ...(tools.length ? { tools } : {}),
      ...extra,
    };

    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal,
      });
    } catch (e) {
      if (signal?.aborted) throw new ProviderError("aborted", 0, false);
      lastErr = new ProviderError(`network error: ${(e as Error).message}`, 0, true);
      continue;
    }

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      // Some providers 400 on stream_options — drop it once and retry immediately.
      if (res.status === 400 && streamOptionsOk && /stream_options/i.test(text)) {
        streamOptionsOk = false;
        attempt--;
        continue;
      }
      const retryable = res.status === 429 || res.status >= 500;
      if (retryable) {
        const ra = res.headers.get("retry-after");
        if (ra && Number.isFinite(Number(ra))) retryAfterMs = Number(ra) * 1000;
        lastErr = new ProviderError(`HTTP ${res.status}: ${trunc(text)}`, res.status, true);
        continue;
      }
      throw new ProviderError(`HTTP ${res.status}: ${trunc(text)}`, res.status, false);
    }

    if (!res.body) {
      lastErr = new ProviderError("empty response body", 0, true);
      continue;
    }
    // Once we start consuming the stream we no longer retry HERE — the agent
    // loop owns mid-stream retries (tagged via midStream).
    try {
      return await consumeSSE(res.body, onText, signal, onReasoning, idleTimeoutMs);
    } catch (e) {
      if (signal?.aborted) throw new ProviderError("aborted", 0, false);
      // Nothing was emitted, so retrying cannot duplicate output: retry HERE
      // rather than surfacing a spurious empty turn.
      if (e instanceof StreamEmptyError) { lastErr = new ProviderError(e.message, 0, true); continue; }
      if (e instanceof StreamStallError) throw new ProviderError(e.message, 0, true, true);
      throw new ProviderError(`stream failed: ${(e as Error).message}`, 0, true, true);
    }
  }
  throw lastErr ?? new ProviderError("request failed", 0, true);
}

/** OpenRouter-style Anthropic cache breakpoints: opt-in per provider. */
function withCacheControl(provider: ResolvedProvider, messages: Msg[]): Msg[] {
  if (!provider.cacheControl) return messages;
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === "user") { lastUserIdx = i; break; }
  }
  return messages.map((m, i) => {
    const mark =
      (m.role === "system" && i === 0) || (m.role === "user" && i === lastUserIdx);
    if (!mark || typeof (m as any).content !== "string") return m;
    return {
      ...m,
      content: [{ type: "text", text: (m as any).content, cache_control: { type: "ephemeral" } }],
    } as Msg;
  });
}

function trunc(s: string, n = 500): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((res, rej) => {
    const t = setTimeout(res, ms);
    signal?.addEventListener(
      "abort",
      () => { clearTimeout(t); rej(new ProviderError("aborted", 0, false)); },
      { once: true },
    );
  });
}
