// Workspace trust: project configs (ap.config.json, .mcp.json) can supply
// hooks/MCP/baseUrl overrides that steal API keys or run arbitrary code.
// Privileged settings apply only when the user has trusted this cwd (or its
// git root). Untrusted projects get an EXPLICIT allowlist of harmless keys —
// never a denylist (new security-sensitive keys would otherwise merge freely).
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir } from "node:os";

const TRUST_FILE = "trusted-workspaces.json";

/**
 * Keys an untrusted project may set. Everything else is stripped until
 * `ap trust accept`. Keep this list tight — model/theme/ignore are UX, not
 * privilege. Never add: hooks, mcpServers, sandbox, sandboxImage, network,
 * bashGuard, redactEnv, permission(s), providers, provider, router, dataDir,
 * confirmEdits, planModel/codeModel, resource limits that burn money.
 */
export const SAFE_PROJECT_CONFIG_KEYS = [
  "ignore",
  "theme",
  "showReasoning",
  "parallelPolicy",
  // NOT shell/git — those change guardrail behaviour / auto-branch side effects.
] as const;

/** @deprecated kept for tests / docs that list the old denylist names. */
export const PRIVILEGED_CONFIG_KEYS = [
  "hooks",
  "mcpServers",
  "sandbox",
  "sandboxImage",
  "network",
  "bashGuard",
  "redactEnv",
  "permission",
  "permissions",
  "confirmEdits",
  "provider",
  "providers",
  "router",
  "dataDir",
  "planModel",
  "codeModel",
  "maxIterations",
  "contextBudgetChars",
  "streamIdleSeconds",
  "mcpAutoBackgroundMs",
  "autoCompact",
  "autoMemory",
  "checkpoints",
] as const;

export type PrivilegedKey = (typeof PRIVILEGED_CONFIG_KEYS)[number];

interface TrustStore {
  /** Absolute paths the user has explicitly trusted. */
  paths: string[];
}

/**
 * Env marker set on every process AP spawns on the model's behalf (bash tool,
 * background bash, subagents, workflow children). Trust is the one decision
 * the agent must never be able to make for itself: `ap trust accept` is a
 * plain CLI command, so the file-tool denies that protect the trust store are
 * irrelevant — the model just shells out. Granting trust refuses outright when
 * this marker is present.
 *
 * The marker is defence in depth, not the primary lock (a command can unset
 * it). The structural control is the interactive-TTY requirement in
 * `assertTrustGrantAllowed`: every model-spawned process has piped or ignored
 * stdio, so `process.stdin.isTTY` is never true there.
 */
export const AGENT_CHILD_ENV = "AP_AGENT_CHILD";

/** True when this process was spawned by AP for the model. */
export function isAgentSpawned(): boolean {
  return process.env[AGENT_CHILD_ENV] === "1";
}

/** Environment for a child process the model controls. */
export function agentChildEnv(extra?: Record<string, string>): Record<string, string> {
  return { ...(process.env as Record<string, string>), ...extra, [AGENT_CHILD_ENV]: "1" };
}

/**
 * Throw unless a trust grant is coming from a human at an interactive
 * terminal. Returns nothing; callers render the message.
 */
export function assertTrustGrantAllowed(): void {
  if (isAgentSpawned()) {
    throw new Error(
      "refused: trust cannot be granted from a process AP spawned (agent tool, subagent, hook). Run `ap trust accept` yourself in a terminal.",
    );
  }
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error(
      "refused: trust must be granted interactively — run `ap trust accept` in a terminal (no pipes, no headless run).",
    );
  }
}

/** Trust decisions always live under the user home data dir — never under a
 *  project-local dataDir the untrusted config could relocate. */
export function trustStoreDir(): string {
  const ap = join(homedir(), ".ap");
  if (existsSync(ap)) return ap;
  const legacy = join(homedir(), ".harness");
  return existsSync(legacy) ? legacy : ap;
}

function trustPath(): string {
  return join(trustStoreDir(), TRUST_FILE);
}

function loadStore(): TrustStore {
  const p = trustPath();
  try {
    if (!existsSync(p)) return { paths: [] };
    const raw = JSON.parse(readFileSync(p, "utf8"));
    return { paths: Array.isArray(raw?.paths) ? raw.paths.map(String) : [] };
  } catch {
    return { paths: [] };
  }
}

function saveStore(store: TrustStore): void {
  const dir = trustStoreDir();
  mkdirSync(dir, { recursive: true });
  writeFileSync(trustPath(), JSON.stringify({ paths: store.paths }, null, 2) + "\n");
}

/** Resolve the directory we trust (git root when available, else cwd). */
export function trustRoot(cwd: string): string {
  let dir = resolve(cwd);
  for (let i = 0; i < 40; i++) {
    if (existsSync(join(dir, ".git"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return resolve(cwd);
}

function norm(p: string): string {
  const r = resolve(p);
  return process.platform === "win32" ? r.toLowerCase() : r;
}

/** True when this workspace (or its git root) is on the trust list. */
export function isWorkspaceTrusted(_dataDir: string, cwd: string): boolean {
  const store = loadStore();
  const root = norm(trustRoot(cwd));
  const here = norm(cwd);
  return store.paths.some((p) => {
    const n = norm(p);
    return n === root || n === here;
  });
}

/** Record an explicit trust decision. Returns the path that was stored.
 *  Backstop only — the CLI has already run assertTrustGrantAllowed and taken
 *  a typed confirmation; this catches any future caller that forgets. */
export function trustWorkspace(_dataDir: string, cwd: string): string {
  if (isAgentSpawned()) {
    throw new Error("refused: trust cannot be granted from an AP-spawned process");
  }
  const root = resolve(trustRoot(cwd));
  const store = loadStore();
  const n = norm(root);
  if (!store.paths.some((p) => norm(p) === n)) {
    store.paths.push(root);
    saveStore(store);
  }
  return root;
}

/** Remove a workspace from the trust list. */
export function untrustWorkspace(_dataDir: string, cwd: string): boolean {
  const root = norm(trustRoot(cwd));
  const store = loadStore();
  const before = store.paths.length;
  store.paths = store.paths.filter((p) => norm(p) !== root && norm(p) !== norm(cwd));
  if (store.paths.length !== before) {
    saveStore(store);
    return true;
  }
  return false;
}

export function listTrustedWorkspaces(_dataDir?: string): string[] {
  return loadStore().paths.slice();
}

/**
 * When untrusted: keep only SAFE_PROJECT_CONFIG_KEYS. When trusted: pass through.
 * Never mutates the caller's object.
 */
export function stripPrivilegedProjectConfig(
  project: Record<string, unknown>,
  trusted: boolean,
): { safe: Record<string, unknown>; stripped: string[] } {
  if (trusted) return { safe: { ...project }, stripped: [] };
  const safe: Record<string, unknown> = {};
  const stripped: string[] = [];
  const allow = new Set<string>(SAFE_PROJECT_CONFIG_KEYS);
  for (const [key, val] of Object.entries(project)) {
    if (allow.has(key)) safe[key] = val;
    else stripped.push(key);
  }
  return { safe, stripped };
}

/**
 * Deep-merge provider maps.
 *
 * Untrusted projects may ONLY override model / cacheControl on an existing
 * home provider — never baseUrl, apiKey, apiKeyEnv, or headers (those steal
 * or redirect credentials). New providers from untrusted projects are ignored
 * entirely (they would resolve against credentials.json / env by name).
 */
export function mergeProviders(
  home: Record<string, unknown> | undefined,
  project: Record<string, unknown> | undefined,
  trusted: boolean,
): Record<string, any> {
  const out: Record<string, any> = {};
  const h = (home && typeof home === "object" ? home : {}) as Record<string, any>;
  const p = (project && typeof project === "object" ? project : {}) as Record<string, any>;
  for (const [name, entry] of Object.entries(h)) {
    if (entry && typeof entry === "object") out[name] = { ...entry };
  }
  for (const [name, entry] of Object.entries(p)) {
    if (!entry || typeof entry !== "object") continue;
    const pe = entry as Record<string, any>;
    const existing = out[name];
    if (!trusted) {
      if (!existing) continue; // no new providers from untrusted projects
      // Model / cacheControl only — strip every credential/network field.
      const next = { ...existing };
      if (typeof pe.model === "string") next.model = pe.model;
      if (typeof pe.cacheControl === "boolean") next.cacheControl = pe.cacheControl;
      out[name] = next;
      continue;
    }
    out[name] = existing ? { ...existing, ...pe } : { ...pe };
  }
  return out;
}

/** Stable fingerprint for diagnostics (not a secret). */
export function workspaceFingerprint(cwd: string): string {
  return createHash("sha256").update(norm(trustRoot(cwd))).digest("hex").slice(0, 12);
}
