// models.dev catalog integration (same database opencode uses).
// Fetched lazily on first use, cached on disk for 24h — never touched on the
// startup hot path.
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Config, ResolvedProvider } from "./config.ts";
import { getKey } from "./creds.ts";

const CATALOG_URL = "https://models.dev/api.json";
const TTL_MS = 24 * 60 * 60 * 1000;

export interface CatalogProvider {
  id: string;
  name?: string;
  api?: string;
  env?: string[];
  npm?: string;
  doc?: string;
  models: Record<string, {
    name?: string;
    cost?: { input?: number; output?: number; cache_read?: number };
    limit?: { context?: number; output?: number };
    reasoning?: boolean;
  }>;
}

export type Catalog = Record<string, CatalogProvider>;

export async function loadCatalog(dataDir: string, force = false): Promise<Catalog> {
  const p = join(dataDir, "models-dev.json");
  if (!force && existsSync(p) && Date.now() - statSync(p).mtimeMs < TTL_MS) {
    try { return JSON.parse(readFileSync(p, "utf8")); } catch {}
  }
  let res: Response;
  try {
    res = await fetch(CATALOG_URL, { signal: AbortSignal.timeout(10_000) });
  } catch (e) {
    if (existsSync(p)) return JSON.parse(readFileSync(p, "utf8")); // offline → stale cache
    throw new Error(`models.dev unreachable: ${(e as Error).message}`);
  }
  if (!res.ok) {
    if (existsSync(p)) return JSON.parse(readFileSync(p, "utf8"));
    throw new Error(`models.dev fetch failed: HTTP ${res.status}`);
  }
  const data = (await res.json()) as Catalog;
  mkdirSync(dataDir, { recursive: true });
  writeFileSync(p, JSON.stringify(data));
  return data;
}

export interface ModelRow {
  provider: string;
  model: string;
  ctx?: number;
  inCost?: number;
  outCost?: number;
}

export function searchModels(catalog: Catalog, query: string, limit = 30): ModelRow[] {
  const q = query.trim().toLowerCase();
  const rows: ModelRow[] = [];
  for (const [pid, prov] of Object.entries(catalog)) {
    for (const [mid, m] of Object.entries(prov.models ?? {})) {
      const hay = `${pid}/${mid} ${m.name ?? ""}`.toLowerCase();
      if (q && !hay.includes(q)) continue;
      rows.push({
        provider: pid,
        model: mid,
        ctx: m.limit?.context,
        inCost: m.cost?.input,
        outCost: m.cost?.output,
      });
      if (rows.length >= limit) return rows;
    }
  }
  return rows;
}

export interface Pricing { input: number; output: number; cacheRead?: number }

/**
 * $/M-token pricing for a model. Exact provider match first; gateways and
 * proxies (not in the db under their own name) fall back to ANY provider
 * listing the same model id — prices for the same model rarely differ much,
 * and the result is labeled approximate everywhere it is shown.
 */
/** USD estimate from token counts + pricing. Pure — safe for unit tests. */
export function estimateUsd(
  pricing: Pricing,
  usage: { prompt: number; cached?: number; completion: number },
): number {
  const cached = usage.cached ?? 0;
  const input = Math.max(0, usage.prompt - cached);
  return (
    (input * pricing.input +
      cached * (pricing.cacheRead ?? pricing.input) +
      usage.completion * pricing.output) / 1e6
  );
}

/** Format a USD estimate for the status /context lines. */
export function formatUsd(usd: number): string {
  if (usd <= 0) return "$0";
  if (usd < 0.01) return `~$${usd.toFixed(4)}`;
  if (usd < 1) return `~$${usd.toFixed(3)}`;
  return `~$${usd.toFixed(2)}`;
}

/** Cache hit rate as a percent string, or "" when there is nothing to report. */
export function cacheHitPct(prompt: number, cached: number): string {
  if (prompt <= 0 || cached <= 0) return "";
  return `${Math.min(100, Math.round((cached / prompt) * 100))}%`;
}

export function modelPricing(catalog: Catalog, provider: string, model: string): Pricing | null {
  const short = model.split("/").pop()!; // proxy ids like "anthropic/claude-x"
  const probe = (prov?: CatalogProvider): Pricing | null => {
    for (const id of [model, short]) {
      const c = prov?.models?.[id]?.cost;
      if (c?.input != null && c.output != null) {
        return { input: c.input, output: c.output, cacheRead: c.cache_read };
      }
    }
    return null;
  };
  const exact = probe(catalog[provider]);
  if (exact) return exact;
  for (const prov of Object.values(catalog)) {
    const hit = probe(prov);
    if (hit) return hit;
  }
  return null;
}

/** Whether a model advertises reasoning support in models.dev (null = not
 *  listed / unknown). Same exact-then-gateway-fallback probe as pricing. */
export function modelReasoning(catalog: Catalog, provider: string, model: string): boolean | null {
  const short = model.split("/").pop()!;
  const probe = (prov?: CatalogProvider): boolean | null => {
    for (const id of [model, short]) {
      const m = prov?.models?.[id];
      if (m && typeof m.reasoning === "boolean") return m.reasoning;
    }
    return null;
  };
  const exact = probe(catalog[provider]);
  if (exact !== null) return exact;
  for (const prov of Object.values(catalog)) {
    const hit = probe(prov);
    if (hit !== null) return hit;
  }
  return null;
}

/**
 * Chat-completions base URL for a catalog provider. models.dev `api` values
 * vary (some include /v1, some don't) — append /v1 when no version segment
 * is present. Best-effort: assumes an OpenAI-compatible endpoint; users can
 * pin an exact baseUrl in ap.config.json when the heuristic is wrong.
 */
export function providerBaseUrl(prov: CatalogProvider): string | null {
  if (!prov.api) return null;
  const api = prov.api.replace(/\/+$/, "");
  return /\/v\d+($|\/)/.test(api) ? api : `${api}/v1`;
}

/** First configured env var that actually has a value. */
export function envKeyFor(prov: CatalogProvider): string | undefined {
  for (const name of prov.env ?? []) {
    const v = process.env[name];
    if (v) return v;
  }
  return undefined;
}

/**
 * Key resolution for a provider, in the exact order resolveCatalogProvider
 * applies: HARNESS_API_KEY > config apiKey > config apiKeyEnv > models.dev
 * env vars > credential store. Exported so the /model picker's "key set"
 * badge can never disagree with what resolution will actually do — pass the
 * same `cp` (or undefined) the resolver would use.
 */
export function catalogKeyFor(config: Config, providerId: string, cp?: CatalogProvider): string | undefined {
  const entry = config.providers[providerId];
  return process.env.HARNESS_API_KEY ?? entry?.apiKey ??
    (entry?.apiKeyEnv ? process.env[entry.apiKeyEnv] : undefined) ??
    (cp ? envKeyFor(cp) : undefined) ??
    getKey(config.dataDir, providerId);
}

/** Resolve a catalog or configured provider into the shared execution shape. */
export async function resolveCatalogProvider(
  config: Config,
  providerId: string,
  modelId?: string,
): Promise<ResolvedProvider> {
  const entry = config.providers[providerId];
  let catalog: Catalog | null = null;
  let cp: CatalogProvider | undefined;
  if (!entry?.baseUrl) {
    catalog = await loadCatalog(config.dataDir);
    cp = catalog[providerId];
  }
  const baseUrl = entry?.baseUrl?.replace(/\/+$/, "") ?? (cp ? providerBaseUrl(cp) : null);
  if (!baseUrl) {
    throw new Error(`provider "${providerId}" is not configured or listed by models.dev — use ap models ${providerId} or add a providers entry`);
  }
  const apiKey = catalogKeyFor(config, providerId, cp) ?? "";
  if (!apiKey) {
    const envs = cp?.env?.join("/") ?? entry?.apiKeyEnv;
    throw new Error(`no API key for provider "${providerId}" — run: ap auth ${providerId}${envs ? `  (or set ${envs})` : ""}`);
  }
  const models = Object.keys(cp?.models ?? {});
  const model = modelId ?? entry?.model ?? (models.length === 1 ? models[0] : undefined);
  if (!model) {
    const count = models.length;
    throw new Error(`choose a model for "${providerId}" — ${count} catalog models available; run: ap models ${providerId}`);
  }
  return {
    name: providerId,
    baseUrl,
    apiKey,
    model,
    cacheControl: entry?.cacheControl ?? false,
    headers: entry?.headers ?? {},
  };
}
