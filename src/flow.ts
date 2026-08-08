// Dynamic workflows: a workflow is a plain TypeScript/JavaScript file the
// USER writes — determinism lives in the script, intelligence lives in
// bounded agent() calls. Flows are user-invoked only (ap flow <name> or
// /flow in the REPL); there is deliberately NO flow tool, so a model can
// never launch one, and agent() children run --light, so they can never
// spawn further agents. Same trust tier as hooks.onDone: local code the
// user chose to put in their own workflow directory.
//
//   // .ap/workflows/audit.ts
//   export default async function ({ agent, parallel, log, args }) {
//     const files = await agent("list the files changed in the last 5 commits");
//     const findings = await parallel([1, 2].map((n) => () =>
//       agent(`audit part ${n} of: ${files}`, { schema: { type: "object", ... } })));
//     return findings.filter(Boolean);
//   }
import { existsSync, readdirSync } from "node:fs";
import { availableParallelism } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { Config } from "./config.ts";
import { postChannel, readChannel } from "./channels.ts";
import { runSubagent } from "./tools/subagent.ts";

const MAX_FLOW_AGENTS = 50;
const DEFAULT_PARALLEL = Math.max(2, Math.min(8, availableParallelism() - 2));

export interface FlowAgentOpts {
  /** Named agent profile (.ap/agents/<name>.md). */
  name?: string;
  cwd?: string;
  /** Seconds, default 300. */
  timeout?: number;
  /** JSON schema the reply must match. The value is parsed AND shape-checked
   *  (type, required keys, array item types — see validateShape); a failure
   *  triggers one retry that feeds the exact mismatch back to the agent. */
  schema?: object;
}

export interface FlowApi {
  agent: (task: string, opts?: FlowAgentOpts) => Promise<any>;
  parallel: <T>(thunks: (() => Promise<T>)[], cap?: number) => Promise<(T | null)[]>;
  log: (msg: string) => void;
  args: string[];
  channel: {
    post: (id: string, text: string, from?: string) => boolean;
    read: (id: string, limit?: number) => { at: string; from: string; text: string }[];
  };
}

/** Injectable for tests: what one agent() call actually does. */
export type FlowRunner = (task: string, opts: { cwd: string; timeoutMs: number; extraArgs: string[] }) => Promise<string>;

/** Resolve a flow name to a file: explicit path > .ap/workflows > dataDir. */
export function resolveFlowPath(config: Config, name: string): string | null {
  if (/\.(ts|js|mjs)$/.test(name)) {
    const p = resolve(config.cwd, name);
    return existsSync(p) ? p : null;
  }
  if (!/^[\w-]+$/.test(name)) return null; // names are identifiers, not paths
  for (const dir of [join(config.cwd, ".ap", "workflows"), join(config.dataDir, "workflows")]) {
    for (const ext of ["ts", "js", "mjs"]) {
      const p = join(dir, `${name}.${ext}`);
      if (existsSync(p)) return p;
    }
  }
  return null;
}

/** Discover runnable workflow scripts (name without extension). */
export function listFlows(config: Config): { name: string; path: string }[] {
  const seen = new Set<string>();
  const out: { name: string; path: string }[] = [];
  for (const dir of [join(config.cwd, ".ap", "workflows"), join(config.dataDir, "workflows")]) {
    try {
      for (const f of readdirSync(dir).sort()) {
        const m = f.match(/^([\w-]+)\.(ts|js|mjs)$/);
        if (!m) continue;
        const name = m[1]!;
        if (seen.has(name)) continue;
        seen.add(name);
        out.push({ name, path: join(dir, f) });
      }
    } catch {}
  }
  return out;
}

/** Concurrency-capped parallel: all thunks settle; failures become null. */
export async function boundedParallel<T>(thunks: (() => Promise<T>)[], cap = DEFAULT_PARALLEL): Promise<(T | null)[]> {
  const results: (T | null)[] = new Array(thunks.length).fill(null);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= thunks.length) return;
      try { results[i] = await thunks[i]!(); } catch { results[i] = null; }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(cap, thunks.length)) }, worker));
  return results;
}

/** Strip ```fences``` and find the outermost JSON value in a model reply. */
export function extractJson(text: string): unknown {
  const t = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try { return JSON.parse(t); } catch {}
  const start = t.search(/[[{]/);
  if (start === -1) throw new Error("no JSON found in reply");
  // Walk to the matching close bracket (string-aware).
  const open = t[start]!;
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inStr = false;
  for (let i = start; i < t.length; i++) {
    const ch = t[i]!;
    if (inStr) {
      if (ch === "\\") i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === open) depth++;
    else if (ch === close && --depth === 0) {
      return JSON.parse(t.slice(start, i + 1));
    }
  }
  throw new Error("unbalanced JSON in reply");
}

/**
 * Structural check against the useful subset of JSON Schema: type, required,
 * nested properties, and array items. Deliberately NOT a full validator (that
 * would be a dependency) — it catches the mistakes models actually make
 * (wrong top-level type, missing required key, scalar where an array belongs)
 * and returns a human path like `findings[0].title` so the retry prompt can
 * name the exact problem. Returns null when the value conforms.
 */
export function validateShape(value: unknown, schema: any, path = "value"): string | null {
  if (!schema || typeof schema !== "object") return null;
  const t = schema.type;
  if (t) {
    const ok =
      t === "array" ? Array.isArray(value) :
      t === "object" ? value !== null && typeof value === "object" && !Array.isArray(value) :
      t === "integer" ? typeof value === "number" && Number.isInteger(value) :
      t === "null" ? value === null :
      typeof value === t;
    if (!ok) return `${path} should be ${t}, got ${Array.isArray(value) ? "array" : value === null ? "null" : typeof value}`;
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value as never)) {
    return `${path} should be one of ${JSON.stringify(schema.enum)}, got ${JSON.stringify(value)}`;
  }
  if (t === "object" || schema.properties || schema.required) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return null; // type check above already reported
    const obj = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (!(key in obj)) return `${path}.${key} is required but missing`;
    }
    for (const [key, sub] of Object.entries(schema.properties ?? {})) {
      if (key in obj) {
        const err = validateShape(obj[key], sub, `${path}.${key}`);
        if (err) return err;
      }
    }
  }
  if (Array.isArray(value) && schema.items) {
    for (let i = 0; i < value.length; i++) {
      const err = validateShape(value[i], schema.items, `${path}[${i}]`);
      if (err) return err;
    }
  }
  return null;
}

/**
 * Run a flow. `out` receives progress lines (the caller renders them — REPL
 * prints dim, CLI prints to stderr, so stdout stays clean for the result).
 * Returns whatever the flow's default export returns.
 */
export async function runFlow(
  config: Config,
  nameOrPath: string,
  args: string[] = [],
  out: (line: string) => void = () => {},
  runner?: FlowRunner,
): Promise<unknown> {
  const path = resolveFlowPath(config, nameOrPath);
  if (!path) {
    throw new Error(
      `no workflow "${nameOrPath}" — looked for .ap/workflows/${nameOrPath}.{ts,js,mjs} and ${join(config.dataDir, "workflows")}`,
    );
  }
  const mod = await import(pathToFileURL(path).href);
  const fn = mod.default;
  if (typeof fn !== "function") {
    throw new Error(`${path} must "export default async function ({ agent, parallel, log, args }) { … }"`);
  }

  let spawned = 0;
  const run: FlowRunner = runner ?? (async (task, o) => {
    const r = await runSubagent({ task, cwd: o.cwd, timeoutMs: o.timeoutMs, extraArgs: o.extraArgs });
    if (r.killed) throw new Error("agent timed out");
    if (!r.text && (r.errorMsg || r.exitCode !== 0)) {
      throw new Error(r.errorMsg || r.stderrTail.trim() || `agent exited ${r.exitCode} with no output`);
    }
    return r.text;
  });

  const agent = async (task: string, opts: FlowAgentOpts = {}): Promise<any> => {
    if (typeof task !== "string" || !task.trim()) throw new Error("agent(task) needs a non-empty string");
    if (++spawned > MAX_FLOW_AGENTS) throw new Error(`flow exceeded ${MAX_FLOW_AGENTS} agent calls — split the work or raise the cap in flow.ts`);
    const extraArgs = opts.name ? ["--agent", opts.name] : [];
    const cwd = opts.cwd ? resolve(config.cwd, opts.cwd) : config.cwd;
    const timeoutMs = Math.min((opts.timeout ?? 300) * 1000, 900_000);
    const n = spawned;
    const short = task.replace(/\s+/g, " ").slice(0, 60);
    out(`◇ agent ${n}/${MAX_FLOW_AGENTS} ${short}`);
    const t0 = Date.now();

    const once = async (prompt: string) => run(prompt, { cwd, timeoutMs, extraArgs });
    let text: string;
    if (opts.schema) {
      const ask = `${task}\n\nRespond with ONLY JSON matching this schema — no prose, no code fences:\n${JSON.stringify(opts.schema)}`;
      // Parse AND shape-check; either failure earns exactly one retry that
      // names the specific problem (parse error or the failing field path).
      const parseAndCheck = (raw: string) => {
        const v = extractJson(raw);
        const err = validateShape(v, opts.schema);
        if (err) throw new Error(`schema mismatch: ${err}`);
        return v;
      };
      text = await once(ask);
      try {
        const v = parseAndCheck(text);
        out(`  ✓ agent ${n} · ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        return v;
      } catch (e) {
        out(`  ↻ agent ${n}: ${(e as Error).message.slice(0, 80)} — one retry`);
        text = await once(`${ask}\n\nYour previous reply was rejected (${(e as Error).message}). Reply with ONLY the corrected JSON value.`);
        const v = parseAndCheck(text); // second failure throws to the flow
        out(`  ✓ agent ${n} · ${((Date.now() - t0) / 1000).toFixed(1)}s (retry)`);
        return v;
      }
    }
    text = await once(task);
    out(`  ✓ agent ${n} · ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    return text;
  };

  const api: FlowApi = {
    agent,
    parallel: boundedParallel,
    log: (msg) => out(String(msg)),
    args,
    channel: {
      post: (id, text, from = "flow") => postChannel(config.dataDir, id, from, text),
      read: (id, limit) => readChannel(config.dataDir, id, limit),
    },
  };
  return fn(api);
}
