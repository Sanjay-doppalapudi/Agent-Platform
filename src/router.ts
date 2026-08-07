import type { CliFlags } from "./index.ts";
import { resolveCatalogProvider } from "./models.ts";
import { ProviderError, streamChat, type Msg, type ToolSchema } from "./provider.ts";
import type { Config, ResolvedProvider } from "./config.ts";
import type { AssembledResponse } from "./stream.ts";

export interface RouterConfig {
  targets: string[];
  fallback?: boolean;
}

export type ProviderRoute = ResolvedProvider | ResolvedProvider[];

export async function routeTargets(config: Config, flags: CliFlags): Promise<ResolvedProvider[]> {
  const names = config.router?.targets?.length ? config.router.targets : [flags.provider ?? process.env.HARNESS_PROVIDER ?? config.provider];
  const targets: ResolvedProvider[] = [];
  for (const target of names) {
    const [providerName = "", ...modelParts] = target.split("/");
    const modelOverride = flags.model ?? (modelParts.length ? modelParts.join("/") : undefined);
    if (!providerName) throw new Error(`router target "${target}" has no provider name`);
    const resolved = await resolveCatalogProvider(config, providerName, modelOverride);
    if (flags.apiKey) resolved.apiKey = flags.apiKey;
    targets.push(resolved);
  }
  if (!targets.length) throw new Error("router has no model targets");
  return targets;
}

export async function streamRouted(
  targets: ResolvedProvider[],
  fallback: boolean,
  messages: Msg[],
  tools: ToolSchema[],
  onText: (delta: string) => void,
  signal?: AbortSignal,
  extra?: Record<string, unknown>,
  onReasoning?: (delta: string) => void,
  idleTimeoutMs = 0,
): Promise<AssembledResponse> {
  let last: unknown;
  const candidates = fallback ? targets : targets.slice(0, 1);
  for (let i = 0; i < candidates.length; i++) {
    try {
      return await streamChat(candidates[i]!, messages, tools, onText, signal, extra, onReasoning, idleTimeoutMs);
    } catch (e) {
      last = e;
      const pe = e as ProviderError;
      if (!fallback || !pe.retryable || pe.midStream || i === candidates.length - 1) throw e;
    }
  }
  throw last ?? new Error("router request failed");
}
