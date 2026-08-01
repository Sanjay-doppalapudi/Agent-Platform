// models.dev catalog integration (same database opencode uses).
// Fetched lazily on first use, cached on disk for 24h — never touched on the
// startup hot path.
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

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
    cost?: { input?: number; output?: number };
    limit?: { context?: number; output?: number };
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

/**
 * Chat-completions base URL for a catalog provider. models.dev `api` values
 * vary (some include /v1, some don't) — append /v1 when no version segment
 * is present. Best-effort: assumes an OpenAI-compatible endpoint; users can
 * pin an exact baseUrl in harness.config.json when the heuristic is wrong.
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
