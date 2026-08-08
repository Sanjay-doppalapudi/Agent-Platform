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
    public code?: string,
  ) {
    super(message);
  }
}

/** Parse Retry-After as seconds or HTTP-date; capped. */
export function parseRetryAfterMs(raw: string | null, capMs = 60_000): number | undefined {
  if (!raw) return undefined;
  if (Number.isFinite(Number(raw))) return Math.min(Number(raw) * 1000, capMs);
  const when = Date.parse(raw);
  if (!Number.isFinite(when)) return undefined;
  return Math.min(Math.max(0, when - Date.now()), capMs);
}

// Providers that reject stream_options get it dropped for that provider only.
const streamOptionsOk = new Map<string, boolean>();

function providerKey(provider: ResolvedProvider): string {
  return `${provider.name}\u0000${provider.baseUrl}`;
}

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

    const key = providerKey(provider);
    const supportsStreamOptions = streamOptionsOk.get(key) ?? true;
    const body: Record<string, unknown> = {
      ...(extra ?? {}),
      model: provider.model,
      messages: withCacheControl(provider, messages),
      stream: true,
      ...(supportsStreamOptions ? { stream_options: { include_usage: true } } : {}),
      ...(tools.length ? { tools } : {}),
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
      if (res.status === 400 && supportsStreamOptions && /stream_options/i.test(text)) {
        streamOptionsOk.set(key, false);
        attempt--;
        continue;
      }
      const retryable = res.status === 429 || res.status >= 500;
      if (retryable) {
        retryAfterMs = parseRetryAfterMs(res.headers.get("retry-after"), 60_000);
        lastErr = new ProviderError(
          `HTTP ${res.status}: ${trunc(text)}`,
          res.status,
          true,
          false,
          res.status === 429 ? "rate_limit" : undefined,
        );
        continue;
      }
      throw new ProviderError(`HTTP ${res.status}: ${trunc(text)}`, res.status, false);
    }

    if (!res.body) {
      lastErr = new ProviderError("empty response body", 0, true);
      continue;
    }
    // `midStream` must mean "text already reached the user", because that is
    // the only case a retry can duplicate. Count emitted characters: a failure
    // before the first content token is retried HERE, invisibly, with no
    // duplication — that covers the common stall/dropped-connection case.
    let emitted = 0;
    const tap = (d: string) => { emitted += d.length; onText(d); };
    try {
      return await consumeSSE(res.body, tap, signal, onReasoning, idleTimeoutMs);
    } catch (e) {
      if (signal?.aborted) throw new ProviderError("aborted", 0, false);
      const msg = e instanceof StreamStallError || e instanceof StreamEmptyError
        ? e.message
        : `stream failed: ${(e as Error).message}`;
      if (emitted === 0) { lastErr = new ProviderError(msg, 0, true); continue; }
      // Text is already on a pipe / in the editor / in the SSE feed and cannot
      // be retracted — hand it to the agent loop, which marks the boundary.
      throw new ProviderError(msg, 0, true, true);
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
