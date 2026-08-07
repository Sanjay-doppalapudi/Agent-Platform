// One-shot headless mode: `ap run -p "task" [--json]`
// --json → one AgentEvent per line (NDJSON) on stdout.
import { loadConfig, resolveProvider } from "./config.ts";
import { routeTargets } from "./router.ts";
import { lifecycleSettled, runTurn, type AgentEvent } from "./agent.ts";
import { buildSystemPrompt } from "./prompt.ts";
import { Checkpoints } from "./checkpoint.ts";
import { initMcp } from "./mcp.ts";
import { getTool } from "./tools/index.ts";
import { MdRenderer } from "./md.ts";
import { errorHint, renderDiff, toolLabel, toolSummary } from "./ui.ts";
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

  // Named agent profile: role instructions + optional model/tool whitelist.
  let agentDef: import("./agents.ts").AgentDef | null = null;
  if (flags.agent) {
    const { discoverAgents } = await import("./agents.ts");
    const defs = discoverAgents(config);
    agentDef = defs.find((a) => a.name === flags.agent!.toLowerCase()) ?? null;
    if (!agentDef) {
      console.error(`unknown agent "${flags.agent}" — available: ${defs.map((a) => a.name).join(", ") || "(none — add .ap/agents/<name>.md)"}`);
      process.exit(1);
    }
    if (agentDef.tools?.length) config.toolFilter = agentDef.tools;
    if (agentDef.model && !flags.model) {
      // "provider/model" only when the prefix is a configured provider —
      // model ids themselves contain slashes (e.g. anthropic/claude-…).
      const slash = agentDef.model.indexOf("/");
      const pfx = slash > 0 ? agentDef.model.slice(0, slash) : "";
      if (pfx && config.providers[pfx]) {
        flags.provider = pfx;
        flags.model = agentDef.model.slice(slash + 1);
      } else {
        flags.model = agentDef.model;
      }
    }
  }

  const provider = config.router?.targets?.length
    ? await routeTargets(config, flags)
    : resolveProvider(config, flags);
  const session = flags.session
    ? Session.load(config.dataDir, flags.session)
    : Session.create(config.dataDir, { cwd: config.cwd, model: Array.isArray(provider) ? provider[0]!.model : provider.model, at: new Date().toISOString() });
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
      case "error": {
        process.stderr.write(`error: ${e.message}\n`);
        const hint = errorHint(e.message);
        if (hint) process.stderr.write(`\x1b[2m${hint}\x1b[0m\n`);
        break;
      }
    }
  };

  const ctrl = new AbortController();
  process.on("SIGINT", () => ctrl.abort());

  await initMcp(config, (m) => process.stderr.write(`\x1b[33m⚠ ${m}\x1b[0m\n`));

  try {
    const { parseEffort } = await import("./theme.ts");
    const level = parseEffort(flags.effort ?? config.reasoningEffort ?? "");
    const effort = level && level !== "off" ? level : undefined;
    await runTurn(config, provider, session, flags.prompt, emit, ctrl.signal, {
      systemOverride:
        flags.system ??
        (agentDef ? `${buildSystemPrompt(config)}\n\nRole — you are the "${agentDef.name}" agent:\n${agentDef.body}` : undefined),
      permit: flags.allowOutside ? async () => true : undefined, // undefined → auto-deny
      extra: effort ? { reasoning_effort: effort } : undefined,
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
    // Background tasks are part of the run's work, and process.exit would
    // kill their children on Windows — wait like lifecycleSettled does, then
    // REPORT their results: a one-shot run has no next turn to fold notes
    // into, so anything not surfaced here is silently lost work.
    await settleAndReportTasks(json);
    await lifecycleSettled();
    // A cancelled run is not a successful run — CI and wrapper scripts read
    // this exit code (130 = terminated by SIGINT, per shell convention).
    process.exit(ctrl.signal.aborted ? 130 : 0);
  } catch (e) {
    if (!json) console.error(`\nfailed: ${(e as Error).message}`);
    else process.stdout.write(JSON.stringify({ type: "error", message: (e as Error).message, retryable: false }) + "\n");
    // The failure path must settle too — otherwise process.exit(1) kills
    // background children on Windows and their work vanishes unreported.
    await settleAndReportTasks(json);
    await lifecycleSettled();
    process.exit(1);
  }
}

/** Wait for background tasks, then print what they produced (NDJSON event in
 *  --json mode, dim text otherwise). Never throws. */
async function settleAndReportTasks(json: boolean): Promise<void> {
  try {
    const { tasksRunning, settleTasks, drainTaskNotes } = await import("./tasks.ts");
    if (tasksRunning()) {
      if (!json) process.stderr.write("\x1b[2mwaiting for background tasks…\x1b[0m\n");
      await settleTasks();
    }
    for (const note of drainTaskNotes()) {
      if (json) process.stdout.write(JSON.stringify({ type: "task_result", text: note }) + "\n");
      else process.stderr.write(`\x1b[2m${note}\x1b[0m\n`);
    }
  } catch {}
}
