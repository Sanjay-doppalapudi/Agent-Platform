// Config loading + provider resolution.
// Precedence: CLI flags > env vars > ./harness.config.json (walked upward) > ~/.harness/config.json
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
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
  permissions: "yolo" | "prompt";
  maxIterations: number;
  contextBudgetChars: number;
  redactEnv: boolean;
  shell: "auto" | "bash" | "powershell" | "cmd";
  parallelPolicy: "safe" | "all" | "none";
  ignore: string[];
  dataDir: string;
  cwd: string;
}

export interface ResolvedProvider {
  name: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  cacheControl: boolean;
  headers: Record<string, string>;
}

const DEFAULTS: Omit<Config, "provider" | "providers" | "cwd"> = {
  permissions: "yolo",
  maxIterations: 40,
  contextBudgetChars: 400_000,
  redactEnv: true,
  shell: "auto",
  parallelPolicy: "safe",
  ignore: [],
  dataDir: join(homedir(), ".harness"),
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

/** Walk upward from cwd looking for harness.config.json. */
function findProjectConfig(startDir: string): Record<string, unknown> | null {
  let dir = resolve(startDir);
  for (let i = 0; i < 20; i++) {
    const found = readJson(join(dir, "harness.config.json"));
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
  const home = readJson(join(homedir(), ".harness", "config.json")) ?? {};
  const project = findProjectConfig(cwd) ?? {};
  const merged = { ...DEFAULTS, provider: "", providers: {}, ...home, ...project } as unknown as Config;
  merged.cwd = cwd;
  merged.dataDir = expandHome(merged.dataDir ?? DEFAULTS.dataDir);
  merged.providers ??= {};
  merged.ignore ??= [];
  if (flags.provider) merged.provider = flags.provider;
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
      `Set one in harness.config.json or pass --base-url/--api-key.`,
    );
  }
  const apiKey =
    flags.apiKey ??
    process.env.HARNESS_API_KEY ??
    entry.apiKey ??
    (entry.apiKeyEnv ? process.env[entry.apiKeyEnv] : undefined) ??
    "";
  if (!apiKey) {
    throw new Error(
      `no API key for provider "${name}" (set ${entry.apiKeyEnv ?? "apiKey in config"} or --api-key)`,
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
