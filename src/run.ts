// One-shot headless mode: `ap run -p "task" [--json]`
// --json → one AgentEvent per line (NDJSON) on stdout.
import { loadConfig, resolveProvider } from "./config.ts";
import { lifecycleSettled, runTurn, type AgentEvent } from "./agent.ts";
import { Checkpoints } from "./checkpoint.ts";
import { initMcp } from "./mcp.ts";
import { getTool } from "./tools/index.ts";
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
  if (config.permissions === "prompt") {
    config.permissions = "yolo";
    process.stderr.write(`permissions:"prompt" is interactive-only — running as yolo (sandbox still applies)\n`);
  }
  const provider = resolveProvider(config, flags);
  const session = flags.session
    ? Session.load(config.dataDir, flags.session)
    : Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
  config.sessionId = session.id;

  const json = !!flags.json;
  const md = !json && process.stdout.isTTY ? new MdRenderer() : null;
  const active = new Map<string, string>(); // running tools: id → label
  let sawText = false;
  let mutated = false;
  const emit = (e: AgentEvent) => {
    if (json) {
      if (e.type === "tool_end" && !e.error && getTool(e.name)?.readOnly === false) mutated = true;
      process.stdout.write(JSON.stringify(e) + "\n");
      return;
    }
    switch (e.type) {
      case "reasoning":
        process.stderr.write(`\x1b[2m${e.delta}\x1b[0m`); // dim, stderr — keeps stdout pipeable
        break;
      case "warn":
        process.stderr.write(`\x1b[33m⚠ ${e.message}\x1b[0m\n`);
        break;
      case "subline":
        process.stderr.write(`\x1b[2m  ${e.text}\x1b[0m\n`);
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
        if (!e.error && getTool(e.name)?.readOnly === false) mutated = true;
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

  await initMcp(config, (m) => process.stderr.write(`\x1b[33m⚠ ${m}\x1b[0m\n`));

  try {
    await runTurn(config, provider, session, flags.prompt, emit, ctrl.signal, {
      systemOverride: flags.system,
      permit: flags.allowOutside ? async () => true : undefined, // undefined → auto-deny
    });
    if (md) process.stdout.write(md.flush());
    if (mutated) {
      const cp = new Checkpoints(config, session.id);
      if (cp.available()) {
        const hash = cp.commit(flags.prompt);
        if (hash && !json) process.stderr.write(`\x1b[2m✓ checkpoint ${hash}\x1b[0m\n`);
      }
    }
    if (!json) process.stdout.write("\n");
    await lifecycleSettled();
    process.exit(0);
  } catch (e) {
    if (!json) console.error(`\nfailed: ${(e as Error).message}`);
    else process.stdout.write(JSON.stringify({ type: "error", message: (e as Error).message, retryable: false }) + "\n");
    await lifecycleSettled();
    process.exit(1);
  }
}
