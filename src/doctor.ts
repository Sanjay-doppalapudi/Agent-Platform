// `ap doctor` — environment diagnostics. Answers "why doesn't it work?" without
// a support round-trip: checks the things that actually break installs (missing
// ripgrep, unresolvable provider key, unreachable endpoint, dead MCP servers,
// data-dir permissions) and prints a fix for each failure. Pure checks, no LLM
// call except an explicit --probe. Never on the startup path.
import { accessSync, constants, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { resolveProvider, type Config, type ResolvedProvider } from "./config.ts";
import { resolveCatalogProvider } from "./models.ts";

export type CheckState = "ok" | "warn" | "fail";

export interface Check {
  name: string;
  state: CheckState;
  detail: string;
  /** Actionable next step — only when state is not "ok". */
  fix?: string;
}

const ok = (name: string, detail: string): Check => ({ name, state: "ok", detail });
const warn = (name: string, detail: string, fix: string): Check => ({ name, state: "warn", detail, fix });
const fail = (name: string, detail: string, fix: string): Check => ({ name, state: "fail", detail, fix });

/** Worst state present — drives the exit code (fail → 1). */
export function overallState(checks: Check[]): CheckState {
  if (checks.some((c) => c.state === "fail")) return "fail";
  if (checks.some((c) => c.state === "warn")) return "warn";
  return "ok";
}

function version(cmd: string, args: string[]): string | null {
  try {
    const p = Bun.spawnSync([cmd, ...args], { stdout: "pipe", stderr: "pipe" });
    if (p.exitCode !== 0) return null;
    const out = (p.stdout?.toString() ?? "").trim().split("\n")[0] ?? "";
    return out || "(installed)";
  } catch {
    return null;
  }
}

function checkRipgrep(): Check {
  const v = Bun.which("rg") ? version("rg", ["--version"]) : null;
  return v
    ? ok("ripgrep", v)
    : fail(
        "ripgrep",
        "not found on PATH — grep and glob tools cannot run",
        "winget install BurntSushi.ripgrep.MSVC   ·   brew install ripgrep   ·   apt install ripgrep",
      );
}

function checkGit(): Check {
  const v = Bun.which("git") ? version("git", ["--version"]) : null;
  return v
    ? ok("git", v)
    : warn("git", "not found — checkpoints, /undo, /worktree and /commit are disabled", "install git, or set \"checkpoints\": \"off\" to silence this");
}

function checkDataDir(config: Config): Check {
  const dir = config.dataDir;
  try {
    mkdirSync(dir, { recursive: true });
    const probe = join(dir, `.doctor-${process.pid}`);
    writeFileSync(probe, "ok");
    rmSync(probe, { force: true });
    return ok("data dir", dir);
  } catch (e) {
    return fail("data dir", `${dir} is not writable — ${(e as Error).message}`, "fix the directory's permissions, or set \"dataDir\" in your config to a writable path");
  }
}

function checkCredentials(config: Config): Check {
  const p = join(config.dataDir, "credentials.json");
  if (!existsSync(p)) return ok("credentials", "no stored keys (using env vars or config)");
  try {
    const raw = readFileSync(p, "utf8");
    const names = Object.keys(JSON.parse(raw) as Record<string, unknown>);
    let mode = "";
    if (process.platform !== "win32") {
      const m = statSync(p).mode & 0o777;
      if (m & 0o077) {
        return warn("credentials", `${names.length} key(s), but mode is ${m.toString(8)} — readable by others`, `chmod 600 ${p}`);
      }
      mode = ` · mode ${m.toString(8)}`;
    }
    accessSync(p, constants.R_OK);
    return ok("credentials", `${names.length} key(s): ${names.join(", ")}${mode}`);
  } catch (e) {
    return fail("credentials", `unreadable or corrupt: ${(e as Error).message}`, `delete ${p} and re-run: ap auth <provider>`);
  }
}

function checkProvider(config: Config): Check {
  const names = Object.keys(config.providers);
  if (!names.length) {
    return fail("provider", "none configured", 'add a "providers" block to ap.config.json (see ap.config.example.json), then: ap auth <name>');
  }
  const active = process.env.HARNESS_PROVIDER || config.provider;
  if (!active) return fail("provider", `configured: ${names.join(", ")} — but none selected`, 'set "provider" in your config, or pass --provider <name>');
  const entry = config.providers[active];
  if (!entry) return fail("provider", `selected "${active}" is not in providers: ${names.join(", ")}`, `set "provider" to one of: ${names.join(", ")}`);
  if (!entry.model) return warn("provider", `${active} has no default model`, `add "model" to the ${active} provider, or always pass -m <model>`);
  return ok("provider", `${active} → ${entry.baseUrl} (${entry.model})`);
}

function checkKey(config: Config): Check {
  const active = process.env.HARNESS_PROVIDER || config.provider;
  const entry = config.providers[active];
  if (!entry) return warn("api key", "no active provider to check", "fix the provider check above first");
  const from =
    process.env.HARNESS_API_KEY ? "HARNESS_API_KEY" :
    entry.apiKey ? "config apiKey" :
    entry.apiKeyEnv && process.env[entry.apiKeyEnv] ? `env ${entry.apiKeyEnv}` :
    null;
  if (from) return ok("api key", `resolved from ${from}`);
  try {
    const stored = JSON.parse(readFileSync(join(config.dataDir, "credentials.json"), "utf8")) as Record<string, string>;
    if (stored[active]) return ok("api key", `resolved from the credential store`);
  } catch {}
  return fail("api key", `no key for "${active}"`, `ap auth ${active}${entry.apiKeyEnv ? `   (or set ${entry.apiKeyEnv})` : ""}`);
}

function checkShell(config: Config): Check {
  try {
    // Imported lazily: resolving the shell caches a module-level value.
    const { shellPrefix } = require("./tools/bash.ts") as typeof import("./tools/bash.ts");
    const prefix = shellPrefix(config.shell);
    return ok("shell", prefix.join(" "));
  } catch (e) {
    return fail("shell", (e as Error).message, 'set "shell" to "powershell" or "cmd" in your config');
  }
}

function checkWorkspace(config: Config): Check {
  // Prove writability with a real write, like checkDataDir: accessSync(W_OK)
  // reports success on Windows for directories a deny-ACE or a read-only
  // share will actually refuse — a false OK is the one answer a diagnostic
  // must never give.
  const probe = join(config.cwd, `.ap-doctor-${process.pid}`);
  try {
    writeFileSync(probe, "ok");
    return ok("workspace", config.cwd);
  } catch (e) {
    return warn("workspace", `${config.cwd} is not writable (${(e as Error).message}) — the agent can read but not edit here`, "run ap from a writable project directory");
  } finally {
    try { rmSync(probe, { force: true }); } catch {}
  }
}

/**
 * Does the key actually WORK? A resolvable key is not a working key: a
 * revoked/blocked key passes every static check and then fails every turn,
 * which is exactly the "all checks passed but nothing works" report this
 * command exists to prevent. Costs one 1-token request.
 */
async function checkAuth(config: Config): Promise<Check> {
  let provider: ResolvedProvider;
  try {
    if (config.router?.targets?.length) {
      const [target] = config.router.targets;
      const slash = target!.indexOf("/");
      provider = await resolveCatalogProvider(
        config,
        slash === -1 ? target! : target!.slice(0, slash),
        slash === -1 ? undefined : target!.slice(slash + 1),
      );
    } else {
      const active = process.env.HARNESS_PROVIDER || config.provider;
      if (!config.providers[active]) {
        provider = await resolveCatalogProvider(config, active, undefined);
      } else {
        provider = resolveProvider(config, {} as any);
      }
    }
  } catch (e) {
    return fail("auth", (e as Error).message, "fix the provider/key checks above first");
  }
  let res: Response;
  try {
    res = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${provider.apiKey}`, ...provider.headers },
      body: JSON.stringify({ model: provider.model, messages: [{ role: "user", content: "hi" }], max_tokens: 1 }),
      signal: AbortSignal.timeout(20_000),
    });
  } catch (e) {
    return warn("auth", `could not reach the provider — ${(e as Error).message}`, "check your network/proxy, then re-run ap doctor");
  }
  if (res.ok) {
    res.body?.cancel();
    return ok("auth", `${provider.name} accepted the key (${provider.model})`);
  }
  const body = (await res.text().catch(() => "")).slice(0, 200);
  if (res.status === 401 || res.status === 403) {
    return fail(
      "auth",
      `${provider.name} REJECTED the key — HTTP ${res.status} ${body}`,
      `the key resolves but the provider refuses it: revoked, out of credits, or the account is blocked. Get a fresh key, then — if it came from an env var — update THAT (an env var overrides "ap auth"): setx ${config.providers[provider.name]?.apiKeyEnv ?? "PROVIDER_API_KEY"} "sk-new-key"  (then open a new terminal)`,
    );
  }
  if (res.status === 429) {
    return warn("auth", `rate limited (HTTP 429) — the key is valid`, "wait a moment, or switch model/provider for now");
  }
  if (res.status === 404) {
    return fail("auth", `HTTP 404 for ${provider.model} at ${provider.baseUrl}`, `the model id or base URL is wrong — ap models ${provider.model.split("/").pop()}`);
  }
  return warn("auth", `HTTP ${res.status} ${body}`, "the provider returned an unexpected status — retry, or check its status page");
}

/** Network reachability of the active provider's base URL (HEAD, 5s). */
async function checkEndpoint(config: Config): Promise<Check> {
  const active = process.env.HARNESS_PROVIDER || config.provider;
  const entry = config.providers[active];
  if (!entry?.baseUrl) return warn("endpoint", "no base URL to probe", "fix the provider check above first");
  try {
    const res = await fetch(entry.baseUrl, { method: "HEAD", signal: AbortSignal.timeout(5000) });
    // Any HTTP answer proves reachability; 404/405 on the base path is normal.
    return ok("endpoint", `${entry.baseUrl} reachable (HTTP ${res.status})`);
  } catch (e) {
    const msg = (e as Error).message;
    return warn("endpoint", `${entry.baseUrl} unreachable — ${msg}`, "check your network/proxy; the base URL may also be wrong");
  }
}

async function checkMcp(config: Config): Promise<Check> {
  const { mcpServerSpecs } = await import("./mcp.ts");
  const specs = mcpServerSpecs(config);
  const names = Object.keys(specs);
  if (!names.length) return ok("mcp", "no servers configured");
  const { initMcp, mcpStatus } = await import("./mcp.ts");
  await initMcp(config, () => {});
  const st = mcpStatus();
  const down = st.filter((s) => !s.ok);
  const tools = st.reduce((n, s) => n + s.tools.length, 0);
  if (down.length) {
    return fail(
      "mcp",
      `${st.length - down.length}/${st.length} servers up (${tools} tools) · down: ${down.map((d) => d.name).join(", ")}`,
      `run: ap mcp   for the per-server error, then fix the command/url in .mcp.json`,
    );
  }
  return ok("mcp", `${st.length} server(s), ${tools} tools`);
}

export async function runChecks(config: Config, opts: { mcp?: boolean; net?: boolean } = {}): Promise<Check[]> {
  const checks: Check[] = [
    checkRipgrep(),
    checkGit(),
    checkShell(config),
    checkDataDir(config),
    checkWorkspace(config),
    checkProvider(config),
    checkKey(config),
    checkCredentials(config),
  ];
  if (opts.net !== false) {
    checks.push(await checkEndpoint(config));
    checks.push(await checkAuth(config)); // the check that would have caught a dead key
  }
  if (opts.mcp !== false && !config.light) checks.push(await checkMcp(config));
  return checks;
}
