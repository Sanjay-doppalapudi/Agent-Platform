// One-shot headless mode: `ap run -p "task" [--json]`
// --json → one AgentEvent per line (NDJSON) on stdout.
import { loadConfig, resolveProvider } from "./config.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { MdRenderer } from "./md.ts";
import { renderDiff, toolLabel, toolSummary } from "./ui.ts";
import { Session } from "./session.ts";
import type { CliFlags } from "./index.ts";

export async function runMain(flags: CliFlags) {
  if (!flags.prompt) {
    console.error(`usage: ap run -p "task" [--json] [--session id]`);
    process.exit(1);
  }
  const config = loadConfig(flags);
  const provider = resolveProvider(config, flags);
  const session = flags.session
    ? Session.load(config.dataDir, flags.session)
    : Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });

  const json = !!flags.json;
  const md = !json && process.stdout.isTTY ? new MdRenderer() : null;
  const active = new Map<string, string>(); // running tools: id → label
  let sawText = false;
  const emit = (e: AgentEvent) => {
    if (json) {
      process.stdout.write(JSON.stringify(e) + "\n");
      return;
    }
    switch (e.type) {
      case "reasoning":
        process.stderr.write(`\x1b[2m${e.delta}\x1b[0m`); // dim, stderr — keeps stdout pipeable
        break;
      case "text":
        process.stdout.write(md ? md.push(e.delta) : e.delta);
        sawText = true;
        break;
      case "tool_start":
        if (sawText) {
          if (md) process.stdout.write(md.flush());
          process.stdout.write("\n");
          sawText = false;
        }
        active.set(e.id, toolLabel(e.name, e.args));
        {
          const diff = renderDiff(e.name, e.args);
          if (diff) process.stderr.write(diff);
        }
        break;
      case "tool_end": {
        const label = active.get(e.id) ?? e.name;
        active.delete(e.id);
        process.stderr.write(`${e.error ? "✗" : "✓"} ${label} · ${toolSummary(e.name, e.output, e.error)} · ${e.ms}ms\n`);
        break;
      }
      case "error":
        process.stderr.write(`error: ${e.message}\n`);
        break;
    }
  };

  const ctrl = new AbortController();
  process.on("SIGINT", () => ctrl.abort());

  try {
    await runTurn(config, provider, session, flags.prompt, emit, ctrl.signal, {
      systemOverride: flags.system,
    });
    if (md) process.stdout.write(md.flush());
    if (!json) process.stdout.write("\n");
    process.exit(0);
  } catch (e) {
    if (!json) console.error(`\nfailed: ${(e as Error).message}`);
    else process.stdout.write(JSON.stringify({ type: "error", message: (e as Error).message, retryable: false }) + "\n");
    process.exit(1);
  }
}
