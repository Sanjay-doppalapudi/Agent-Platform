#!/usr/bin/env bun
// Entry point: parse argv, dispatch to a mode via lazy import so the hot path
// (startup → first prompt) loads only what it needs.

const VERSION = "0.1.0";

export interface CliFlags {
  provider?: string;
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  cwd?: string;
  session?: string;
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
  console.log(`harness ${VERSION} — minimal fast coding agent

Usage:
  harness                      interactive REPL
  harness run -p "task"        one-shot run (--json for NDJSON events)
  harness serve [--port 4141]  HTTP server mode
  harness tool <name> '<json>' run a single tool directly (testing)
  harness prompt [--cwd dir]   print the system prompt used for that directory

Flags:
  --provider <name>   named provider from config
  -m, --model <id>    model override
  --base-url <url>    OpenAI-compatible endpoint (with --api-key, no config needed)
  --api-key <key>     API key override
  --cwd <dir>         working directory for the agent
  --session <id>      attach to session id
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
