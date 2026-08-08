// Config loading + provider resolution.
// Precedence: CLI flags > env vars > ./ap.config.json (walked upward, legacy
// harness.config.json accepted) > <dataDir>/config.json
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { getKey } from "./creds.ts";
import type { CliFlags } from "./index.ts";

export interface ProviderEntry {
  baseUrl: string;
  apiKey?: string;
  apiKeyEnv?: string;
  model: string;
  cacheControl?: boolean;
  headers?: Record<string, string>;
}

/** One MCP server: stdio ({command, args, env}) or Streamable HTTP ({url,
 *  headers}). Same JSON shape as Claude Code's .mcp.json entries. */
export interface McpServerSpec {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  /** "stdio" | "http" — informational; the transport is inferred. */
  type?: string;
}

export type PermissionVerdict = "allow" | "ask" | "deny";

export interface RouterPolicy {
  targets: string[];
  fallback?: boolean;
}

export interface Config {
  provider: string;
  providers: Record<string, ProviderEntry>;
  router?: RouterPolicy;
  mode: "plan" | "code";
  permissions: "yolo" | "prompt";
  /** Per-tool rules (opencode-style), evaluated before the permissions mode:
   *  {"edit": "ask", "fetch": "deny", "mcp_*": "ask",
   *   "bash": {"git push*": "ask", "*": "allow"}}
   *  Tool keys and bash command patterns support * globs. "allow" skips the
   *  ask gate only — the path sandbox and bashGuard still apply. */
  permission?: Record<string, PermissionVerdict | Record<string, PermissionVerdict>>;
  sandbox: "workspace" | "off";
  bashGuard: "on" | "off";
  /** Light profile: smallest possible prompt + no optional features. Every
   *  non-essential feature (current and future) must check this flag. */
  light: boolean;
  /** Shadow-repo checkpoints after mutating turns (off in light profile). */
  checkpoints: "on" | "off";
  /** REPL auto-compaction at 85% of contextBudgetChars (full profile only). */
  autoCompact: "on" | "off";
  /** After compact, extract 0–3 memory cards into the repo-keyed memory dir. */
  autoMemory: "on" | "off";
  /** Soft ms before a blocking MCP tool call backgrounds onto the task queue.
   *  0 = never auto-background. Default 30000. Hard timeout remains 120s. */
  mcpAutoBackgroundMs?: number;
  /** Show model reasoning/thinking in the compact REPL view (default on). */
  showReasoning?: "on" | "off";
  /** Force ask before every edit/write even when permissions are yolo. */
  confirmEdits?: boolean;
  /** Default reasoning effort sent as `reasoning_effort` (models that don't
   *  support it may ignore or reject it — /effort checks models.dev). */
  reasoningEffort?: "minimal" | "low" | "medium" | "high";
  /** Color theme name (see /theme). Ignored when NO_COLOR is set. */
  theme?: string;
  /** Shell hooks: preBash/preWrite/preEdit (nonzero exit blocks the tool),
   *  afterEdit (nonzero output is fed back to the model to self-fix).
   *  Lifecycle: onDone/onError — a shell command (AP_EVENT/AP_PAYLOAD env)
   *  or an http(s) URL that receives a JSON POST; fire-and-forget. */
  hooks?: Record<string, string>;
  /** MCP servers (name → spec) — merged with a project .mcp.json. Full
   *  profile only; connected lazily before the first turn, never at startup. */
  mcpServers?: Record<string, McpServerSpec>;
  streamIdleSeconds: number;
  maxIterations: number;
  contextBudgetChars: number;
  redactEnv: boolean;
  shell: "auto" | "bash" | "powershell" | "cmd";
  parallelPolicy: "safe" | "all" | "none";
  ignore: string[];
  dataDir: string;
  cwd: string;
  /** Set by the front-end once a session exists — scopes the plans folder. */
  sessionId?: string;
  /** Programmatic tool whitelist (named agent profiles) — never from JSON.
   *  Constant per process, so schema bytes stay stable for caching. */
  toolFilter?: string[];
  /** Optional model for plan mode ("provider/model" or bare model id).
   *  Swapped into the live provider on /plan; never enters the system prompt. */
  planModel?: string;
  /** Optional model for code mode. Swapped on /code (and as the default). */
  codeModel?: string;
  /** Real-git hygiene (distinct from shadow checkpoints). */
  git?: {
    /** On first mutating tool call while on a protected branch, create and
     *  switch to `ap/<slug>` so main/master stay untouched. Default off. */
    autoBranch?: boolean;
  };
}

export interface ResolvedProvider {
  name: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  cacheControl: boolean;
  headers: Record<string, string>;
}

/** ~/.ap for fresh installs; existing ~/.harness dirs keep working. */
function defaultDataDir(): string {
  const ap = join(homedir(), ".ap");
  if (existsSync(ap)) return ap;
  const legacy = join(homedir(), ".harness");
  return existsSync(legacy) ? legacy : ap;
}

const DEFAULTS: Omit<Config, "provider" | "providers" | "cwd"> = {
  mode: "code",
  permissions: "yolo",
  sandbox: "workspace",
  bashGuard: "on",
  light: false,
  checkpoints: "on",
  autoCompact: "on",
  autoMemory: "on",
  streamIdleSeconds: 90,
  maxIterations: 40,
  contextBudgetChars: 400_000,
  redactEnv: true,
  shell: "auto",
  parallelPolicy: "safe",
  ignore: [],
  dataDir: defaultDataDir(),
};

function readJson(path: string): Record<string, unknown> | null {
  try {
    if (!existsSync(path)) return null;
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    console.error(`warning: failed to parse ${path}: ${(e as Error).message}`);
    return null;
  }
}

/** Walk upward from cwd looking for ap.config.json (or legacy harness.config.json). */
function findProjectConfig(startDir: string): Record<string, unknown> | null {
  let dir = resolve(startDir);
  for (let i = 0; i < 20; i++) {
    const found = readJson(join(dir, "ap.config.json")) ?? readJson(join(dir, "harness.config.json"));
    if (found) return found;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

function expandHome(p: string): string {
  return p.startsWith("~") ? join(homedir(), p.slice(1)) : p;
}

export function loadConfig(flags: CliFlags): Config {
  const cwd = resolve(flags.cwd ?? process.cwd());
  const home = readJson(join(defaultDataDir(), "config.json")) ?? {};
  const project = findProjectConfig(cwd) ?? {};
  const merged = { ...DEFAULTS, provider: "", providers: {}, ...home, ...project } as unknown as Config;
  merged.cwd = cwd;
  merged.dataDir = expandHome(merged.dataDir ?? DEFAULTS.dataDir);
  merged.providers ??= {};
  merged.ignore ??= [];
  if (flags.provider) merged.provider = flags.provider;
  if (flags.mode === "plan" || flags.mode === "code") merged.mode = flags.mode;
  if (flags.noSandbox) merged.sandbox = "off";
  if (flags.light) merged.light = true;
  return merged;
}

export function resolveProvider(config: Config, flags: CliFlags): ResolvedProvider {
  // Ad-hoc endpoint: --base-url (+ --api-key) works with no config file at all.
  if (flags.baseUrl) {
    return {
      name: "adhoc",
      baseUrl: flags.baseUrl.replace(/\/+$/, ""),
      apiKey: flags.apiKey ?? process.env.HARNESS_API_KEY ?? "",
      model: flags.model ?? process.env.HARNESS_MODEL ?? "",
      cacheControl: false,
      headers: {},
    };
  }

  const name = flags.provider ?? process.env.HARNESS_PROVIDER ?? config.provider;
  const entry = config.providers[name];
  if (!entry) {
    const known = Object.keys(config.providers).join(", ") || "(none configured)";
    throw new Error(
      `provider "${name || "(unset)"}" not found. Known providers: ${known}. ` +
      `Set one in ap.config.json or pass --base-url/--api-key.`,
    );
  }
  const apiKey =
    flags.apiKey ??
    process.env.HARNESS_API_KEY ??
    entry.apiKey ??
    (entry.apiKeyEnv ? process.env[entry.apiKeyEnv] : undefined) ??
    getKey(config.dataDir, name) ??
    "";
  if (!apiKey) {
    throw new Error(
      `no API key for provider "${name}" — run: ap auth ${name}  (or set ${entry.apiKeyEnv ?? "--api-key"})`,
    );
  }
  const model = flags.model ?? process.env.HARNESS_MODEL ?? entry.model;
  if (!model) throw new Error(`no model for provider "${name}" (set model in config or -m)`);
  return {
    name,
    baseUrl: entry.baseUrl.replace(/\/+$/, ""),
    apiKey,
    model,
    cacheControl: entry.cacheControl ?? false,
    headers: entry.headers ?? {},
  };
}

/**
 * Resolve a "provider/model" or bare model id against config, keeping the
 * current provider when the prefix is unknown (model ids often contain
 * slashes — same rule as named-agent overrides in run.ts).
 *
 * Does not throw on a missing API key: the REPL/mode switch still updates
 * name+model; the next API call surfaces the auth error via resolveProvider.
 */
export function resolveModelRef(config: Config, ref: string, current: ResolvedProvider): ResolvedProvider {
  const slash = ref.indexOf("/");
  if (slash <= 0) return { ...current, model: ref };
  const pfx = ref.slice(0, slash);
  const modelId = ref.slice(slash + 1);
  if (!modelId) return { ...current, model: ref };
  if (pfx === current.name) return { ...current, model: modelId };
  const entry = config.providers[pfx];
  if (entry) {
    let apiKey =
      entry.apiKey ??
      (entry.apiKeyEnv ? process.env[entry.apiKeyEnv] : undefined) ??
      "";
    if (!apiKey && config.dataDir) {
      try { apiKey = getKey(config.dataDir, pfx) ?? ""; } catch { /* leave empty */ }
    }
    return {
      name: pfx,
      baseUrl: entry.baseUrl.replace(/\/+$/, ""),
      apiKey,
      model: modelId,
      cacheControl: entry.cacheControl ?? false,
      headers: entry.headers ?? {},
    };
  }
  // Bare "org/model" id on the current provider (OpenRouter-style).
  return { ...current, model: ref };
}

/** Apply planModel/codeModel for the given mode. No-op when unset. */
export function providerForMode(
  config: Config,
  mode: "plan" | "code",
  current: ResolvedProvider,
): ResolvedProvider {
  const ref = mode === "plan" ? config.planModel : config.codeModel;
  if (!ref?.trim()) return current;
  return resolveModelRef(config, ref.trim(), current);
}
