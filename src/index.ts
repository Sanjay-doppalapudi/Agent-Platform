#!/usr/bin/env bun
// Entry point: parse argv, dispatch to a mode via lazy import so the hot path
// (startup → first prompt) loads only what it needs.

const VERSION = "0.1.2";

export interface CliFlags {
  provider?: string;
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  cwd?: string;
  session?: string;
  mode?: string;
  resume?: string;
  continue?: boolean;
  json?: boolean;
  port?: number;
  prompt?: string;
  system?: string;
}

function parseArgs(argv: string[]): { cmd: string; flags: CliFlags; rest: string[] } {
  const flags: CliFlags = {};
  const rest: string[] = [];
  let cmd = "";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    switch (a) {
      case "--provider": flags.provider = argv[++i]; break;
      case "--model": case "-m": flags.model = argv[++i]; break;
      case "--base-url": flags.baseUrl = argv[++i]; break;
      case "--api-key": flags.apiKey = argv[++i]; break;
      case "--cwd": flags.cwd = argv[++i]; break;
      case "--session": flags.session = argv[++i]; break;
      case "--mode": flags.mode = argv[++i]; break;
      case "--plan": flags.mode = "plan"; break;
      case "--resume": flags.resume = argv[++i]; break;
      case "--continue": case "-c": flags.continue = true; break;
      case "--json": flags.json = true; break;
      case "--port": flags.port = Number(argv[++i]); break;
      case "--prompt": case "-p": flags.prompt = argv[++i]; break;
      case "--system": flags.system = argv[++i]; break;
      case "--version": case "-v": console.log(VERSION); process.exit(0);
      case "--help": case "-h": printHelp(); process.exit(0);
      default:
        if (!cmd && !a.startsWith("-")) cmd = a;
        else rest.push(a);
    }
  }
  return { cmd, flags, rest };
}

function printHelp() {
  console.log(`AP (Agent Platform) ${VERSION} — minimal fast coding agent

Usage:
  ap                      interactive REPL
  ap run -p "task"        one-shot run (--json for NDJSON events)
  ap serve [--port 4141]  HTTP server mode
  ap tool <name> '<json>' run a single tool directly (testing)
  ap prompt [--cwd dir]   print the system prompt used for that directory
  ap models [query]       search the models.dev catalog (providers + prices)
  ap auth <provider>      store an API key securely (data dir credentials.json)

Flags:
  --provider <name>   named provider from config
  -m, --model <id>    model override
  --base-url <url>    OpenAI-compatible endpoint (with --api-key, no config needed)
  --api-key <key>     API key override
  --cwd <dir>         working directory for the agent
  --session <id>      attach to session id
  --mode <plan|code>  plan = read-only tools, produce a plan (also: --plan)
  --resume <id>       resume a saved session (REPL)
  -c, --continue      resume most recent session (REPL)
  --json              NDJSON event output (run mode)
  -v, --version       print version`);
}

const { cmd, flags, rest } = parseArgs(process.argv.slice(2));

try {
  await dispatch(cmd, flags, rest);
} catch (e) {
  console.error(`error: ${(e as Error).message}`);
  // Double-click launch: the console window vanishes on exit — hold it open
  // so the error is readable. Piped/scripted runs (not a TTY) are unaffected.
  if (process.stdin.isTTY && process.stdout.isTTY) {
    console.error("\npress Enter to exit...");
    const { createInterface } = await import("node:readline");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    await new Promise<void>((res) => rl.question("", () => { rl.close(); res(); }));
  }
  process.exit(1);
}

async function dispatch(cmd: string, flags: CliFlags, rest: string[]) {
switch (cmd) {
  case "run": {
    const { runMain } = await import("./run.ts");
    await runMain(flags);
    break;
  }
  case "serve": {
    const { serveMain } = await import("./server.ts");
    await serveMain(flags);
    break;
  }
  case "tool": {
    const { toolMain } = await import("./tool-cli.ts");
    await toolMain(flags, rest);
    break;
  }
  case "prompt": {
    const { loadConfig } = await import("./config.ts");
    const { buildSystemPrompt } = await import("./prompt.ts");
    const sys = buildSystemPrompt(loadConfig(flags));
    console.log(sys);
    console.error(`\n[${sys.length} chars, ~${Math.round(sys.length / 4)} tokens]`);
    break;
  }
  case "models": {
    const { loadConfig } = await import("./config.ts");
    const { loadCatalog, searchModels } = await import("./models.ts");
    const config = loadConfig(flags);
    const catalog = await loadCatalog(config.dataDir);
    const rows = searchModels(catalog, rest.join(" "));
    if (!rows.length) { console.log("no matches"); break; }
    for (const r of rows) {
      const cost = r.inCost != null ? `$${r.inCost}/$${r.outCost ?? "?"} per M` : "";
      const ctx = r.ctx ? `${Math.round(r.ctx / 1000)}k ctx` : "";
      console.log(`${r.provider}/${r.model}  ${[ctx, cost].filter(Boolean).join(" · ")}`);
    }
    break;
  }
  case "auth": {
    const { loadConfig } = await import("./config.ts");
    const { listKeys, setKey } = await import("./creds.ts");
    const config = loadConfig(flags);
    const [prov, keyArg] = rest;
    if (!prov) {
      const stored = listKeys(config.dataDir);
      console.log(stored.length ? `stored keys: ${stored.join(", ")}` : "no stored keys — usage: harness auth <provider> [key]");
      break;
    }
    let key = keyArg;
    if (!key) {
      const { emitKeypressEvents } = await import("node:readline");
      emitKeypressEvents(process.stdin);
      const { readSecret } = await import("./input.ts");
      key = await readSecret(`API key for ${prov}: `);
    }
    if (!key) { console.error("no key entered"); process.exit(1); }
    setKey(config.dataDir, prov, key);
    console.log(`stored key for "${prov}" in ${config.dataDir}\\credentials.json (user-only access)`);
    break;
  }
  case "": {
    const { replMain } = await import("./repl.ts");
    await replMain(flags);
    break;
  }
  default:
    console.error(`unknown command: ${cmd} (try --help)`);
    process.exit(1);
}
}
