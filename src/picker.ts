// Interactive pickers for the REPL's /model flow (full profile only — the
// call site gates on !config.light, keeping the frozen --light surface).
// pickFromList is a standalone raw-mode selector: type to filter, ↑/↓ to
// move, Enter selects, Esc cancels. It runs BETWEEN prompts, so it owns its
// own screen block — full redraw each keypress, erased on exit — and never
// touches readLine's frame math. Caller must have run
// readline.emitKeypressEvents(process.stdin) once (repl.ts does at startup).
import { currentTheme, reflowRewind, visibleLen } from "./theme.ts";
import { readSecret } from "./input.ts";
import { setKey } from "./creds.ts";
import {
  catalogKeyFor,
  loadCatalog,
  providerBaseUrl,
  resolveCatalogProvider,
  type Catalog,
  type CatalogProvider,
} from "./models.ts";
import type { Config, ResolvedProvider } from "./config.ts";

const R = "\x1b[0m";
const INV = "\x1b[7m";
const DIM = () => currentTheme().dim;

export interface PickItem<T> {
  label: string;
  hint?: string;
  value: T;
}

const LIST_ROWS = 10;

/**
 * Arrow-key list selector with type-to-filter. Resolves the picked value, or
 * null on Esc / ctrl+c (cancel never kills the REPL — unlike readSecret,
 * cancelling a picker is routine). ANSI discipline: each row is ONE span
 * (inverse for the selection, dim for the rest) closed by a single reset, so
 * no nested-reset bleed is possible.
 */
export function pickFromList<T>(title: string, items: PickItem<T>[], initial = 0): Promise<T | null> {
  return new Promise((resolve) => {
    const stdin = process.stdin;
    if (stdin.isTTY) stdin.setRawMode(true);
    stdin.resume();

    let filter = "";
    let idx = Math.min(Math.max(initial, 0), Math.max(items.length - 1, 0));
    let drawn = 0; // rows currently on screen (rewound before each redraw)
    let lastLens: number[] = []; // visible length per drawn line, for the reflow-aware resize rewind

    const matches = (): PickItem<T>[] => {
      const q = filter.toLowerCase();
      if (!q) return items;
      return items.filter((it) => `${it.label} ${it.hint ?? ""}`.toLowerCase().includes(q));
    };

    const render = (rewindStr?: string) => {
      // Honest width, like input.ts: a floor above the real column count
      // would wrap every row when zoomed in. 8 only guards degenerate PTYs.
      const cols = Math.max(process.stdout.columns ?? 80, 8);
      const width = cols - 1;
      const rows = matches();
      if (idx >= rows.length) idx = Math.max(rows.length - 1, 0);
      const lines: string[] = [];
      lines.push(`${DIM()}${title.slice(0, width)}${R}`);
      lines.push(`  filter› ${filter.slice(0, width - 10)}`);
      const start = Math.max(0, Math.min(idx - (LIST_ROWS - 2), rows.length - LIST_ROWS));
      const end = Math.min(rows.length, start + LIST_ROWS);
      if (start > 0) lines.push(`${DIM()}  ↑ ${start} more${R}`);
      for (let i = start; i < end; i++) {
        const it = rows[i]!;
        // Truncate label and hint on PLAIN text, then color the pieces — a
        // slice can never cut an escape sequence in half.
        const label = it.label.slice(0, width - 2);
        const room = width - 2 - label.length - 2;
        const hint = it.hint && room > 4 ? it.hint.slice(0, room) : "";
        lines.push(
          i === idx
            ? `${INV} ${label}${hint ? `  ${hint}` : ""} ${R}`
            : `  ${label}${hint ? `${DIM()}  ${hint}${R}` : ""}`,
        );
      }
      if (end < rows.length) lines.push(`${DIM()}  ↓ ${rows.length - end} more${R}`);
      if (!rows.length) lines.push(`${DIM()}  (no matches)${R}`);
      lines.push(`${DIM()}  ↑↓ move · type to filter · Enter select · Esc cancel${R}`);
      const rewind = rewindStr ?? (drawn ? `\x1b[${drawn - 1}A\r\x1b[J` : "");
      process.stdout.write(rewind + lines.join("\n"));
      drawn = lines.length;
      lastLens = lines.map(visibleLen);
    };

    const done = (value: T | null) => {
      stdin.removeListener("keypress", onKey);
      process.stdout.removeListener("resize", onResize);
      if (resizeTimer) clearTimeout(resizeTimer);
      if (stdin.isTTY) stdin.setRawMode(false);
      // Erase the whole block; the caller prints the outcome on a clean row.
      if (drawn) process.stdout.write(`\x1b[${drawn - 1}A\r\x1b[J`);
      resolve(value);
    };

    // Zoom/resize: debounced redraw with a reflow-aware rewind (see
    // input.ts) — the terminal rewrapped our old rows, so the way back to
    // the block's first row is recomputed from recorded line lengths.
    let resizeTimer: ReturnType<typeof setTimeout> | undefined;
    const onResize = () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        resizeTimer = undefined;
        if (!lastLens.length) return render();
        const cols = Math.max(process.stdout.columns ?? 80, 8);
        const up = Math.min(reflowRewind(lastLens.slice(0, -1), lastLens[lastLens.length - 1]!, cols), 400);
        render(`${up ? `\x1b[${up}A` : ""}\r\x1b[J`);
      }, 40);
    };

    const onKey = (str: string, key: any) => {
      if (!key) return;
      if (key.ctrl && key.name === "c") return done(null);
      if (key.ctrl || key.meta) return;
      const rows = matches();
      switch (key.name) {
        case "return": case "enter":
          return done(rows.length ? rows[Math.min(idx, rows.length - 1)]!.value : null);
        case "escape":
          return done(null);
        case "up":
          idx = Math.max(0, idx - 1);
          return render();
        case "down":
          idx = Math.min(Math.max(rows.length - 1, 0), idx + 1);
          return render();
        case "pageup":
          idx = Math.max(0, idx - LIST_ROWS);
          return render();
        case "pagedown":
          idx = Math.min(Math.max(rows.length - 1, 0), idx + LIST_ROWS);
          return render();
        case "backspace":
          filter = filter.slice(0, -1);
          idx = 0;
          return render();
      }
      if (typeof str === "string" && str.length > 0 && str >= " ") {
        filter += str;
        idx = 0;
        render();
      }
    };

    stdin.on("keypress", onKey);
    process.stdout.on("resize", onResize);
    render();
  });
}

/**
 * Provider rows for the picker: every models.dev provider with a usable
 * OpenAI-compatible endpoint, merged with config.providers (which surface
 * even when models.dev doesn't list them — e.g. custom base URLs). Sorted
 * keyed-first, then configured, then alphabetically: the list opens on
 * providers that can work right now. Pure — unit-tested directly.
 */
export function buildProviderRows(config: Config, catalog: Catalog): PickItem<string>[] {
  const rows: { pid: string; keyed: boolean; configured: boolean; item: PickItem<string> }[] = [];
  const seen = new Set<string>();
  const push = (pid: string, cp: CatalogProvider | undefined, configured: boolean) => {
    const entry = config.providers[pid];
    // Mirror resolveCatalogProvider: a configured baseUrl means the catalog
    // entry (and its env list) is never consulted during resolution.
    const cpForKey = entry?.baseUrl ? undefined : cp;
    const keyed = !!catalogKeyFor(config, pid, cpForKey);
    const nModels = Object.keys(cp?.models ?? {}).length || (entry?.model ? 1 : 0);
    const bits = [
      cp?.name && cp.name !== pid ? cp.name : "",
      `${nModels} model${nModels === 1 ? "" : "s"}`,
      keyed ? "✓ key" : "needs key",
      configured ? "configured" : "",
    ].filter(Boolean);
    rows.push({ pid, keyed, configured, item: { label: pid, hint: bits.join(" · "), value: pid } });
    seen.add(pid);
  };
  for (const pid of Object.keys(config.providers)) push(pid, catalog[pid], true);
  for (const [pid, cp] of Object.entries(catalog)) {
    if (seen.has(pid)) continue;
    if (!providerBaseUrl(cp)) continue; // no endpoint listed — unusable here
    push(pid, cp, false);
  }
  rows.sort((a, b) =>
    Number(b.keyed) - Number(a.keyed) ||
    Number(b.configured) - Number(a.configured) ||
    a.pid.localeCompare(b.pid));
  return rows.map((r) => r.item);
}

/** Model rows for one provider: catalog models (ctx · $in/$out per M tokens),
 *  or the configured default when models.dev has nothing. Pure. */
export function buildModelRows(cp: CatalogProvider | undefined, configuredModel?: string): PickItem<string>[] {
  const entries = Object.entries(cp?.models ?? {});
  if (!entries.length) {
    return configuredModel ? [{ label: configuredModel, hint: "configured default", value: configuredModel }] : [];
  }
  return entries
    .map(([id, m]) => {
      const ctx = m.limit?.context ? `${Math.round(m.limit.context / 1000)}k` : "";
      const cost = m.cost?.input != null ? `$${m.cost.input}/$${m.cost.output ?? "?"}` : "";
      return { label: id, hint: [ctx, cost].filter(Boolean).join(" · "), value: id };
    })
    .sort((a, b) => a.label.localeCompare(b.label));
}

/**
 * The /model interactive flow: provider list (models.dev + config) → key
 * check (prompt + save to the ACL-locked credential store when missing) →
 * model list → resolved provider. Returns null on cancel at any step.
 */
export async function pickModelInteractive(config: Config, current: ResolvedProvider): Promise<ResolvedProvider | null> {
  const catalog = await loadCatalog(config.dataDir);
  const provRows = buildProviderRows(config, catalog);
  if (!provRows.length) throw new Error("no providers available — models.dev returned nothing and none are configured");
  const preP = Math.max(provRows.findIndex((r) => r.value === current.name), 0);
  const pid = await pickFromList(`select a provider (${provRows.length}, models.dev + config) — current: ${current.name}/${current.model}`, provRows, preP);
  if (!pid) return null;

  const entry = config.providers[pid];
  const cp = catalog[pid];
  if (!catalogKeyFor(config, pid, entry?.baseUrl ? undefined : cp)) {
    const envs = cp?.env?.length ? ` (${cp.env.join(" / ")})` : "";
    const key = (await readSecret(`  API key for ${pid}${envs} — Enter with nothing to cancel: `)).trim();
    if (!key) return null;
    setKey(config.dataDir, pid, key);
    console.log(`${DIM()}  key saved to the credential store (change later with: ap auth ${pid})${R}`);
  }

  const modelRows = buildModelRows(cp, entry?.model);
  if (!modelRows.length) throw new Error(`models.dev lists no models for "${pid}" and no default is configured — /model ${pid}/<model> to name one`);
  let mid: string | null;
  if (modelRows.length === 1) {
    mid = modelRows[0]!.value; // nothing to choose
  } else {
    const cur = pid === current.name ? current.model : entry?.model;
    const preM = Math.max(modelRows.findIndex((r) => r.value === cur), 0);
    mid = await pickFromList(`select a model — ${pid} (${modelRows.length})`, modelRows, preM);
  }
  if (!mid) return null;
  return resolveCatalogProvider(config, pid, mid);
}
