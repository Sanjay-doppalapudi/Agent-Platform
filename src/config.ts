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

export interface Config {
  provider: string;
  providers: Record<string, ProviderEntry>;
  mode: "plan" | "code";
  permissions: "yolo" | "prompt";
  sandbox: "workspace" | "off";
  bashGuard: "on" | "off";
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
