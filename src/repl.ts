// Interactive REPL: streaming render, Ctrl+C aborts the turn (not the
// process), :commands for session control.
import { createInterface } from "node:readline";
import { loadConfig, resolveProvider } from "./config.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { buildSystemPrompt } from "./prompt.ts";
import { MdRenderer } from "./md.ts";
import { renderDiff, toolLabel, toolSummary } from "./ui.ts";
import { Session } from "./session.ts";
import type { Usage } from "./stream.ts";
import type { CliFlags } from "./index.ts";

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;

function makeSpinner() {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let timer: ReturnType<typeof setInterval> | null = null;
  let i = 0;
  let label = "thinking";
  return {
    start(newLabel = "thinking") {
      label = newLabel;
      if (timer || !process.stdout.isTTY) return;
      timer = setInterval(() => {
        process.stdout.write(`\r${dim(`${frames[i++ % frames.length]} ${label}`)}\x1b[K`);
      }, 80);
    },
    stop() {
      if (!timer) return;
      clearInterval(timer);
      timer = null;
      process.stdout.write("\r\x1b[K");
    },
  };
}

export async function replMain(flags: CliFlags) {
  const config = loadConfig(flags);
  let provider = resolveProvider(config, flags);

  let session: Session;
  if (flags.resume) {
    session = Session.load(config.dataDir, flags.resume);
  } else if (flags.continue) {
    const latest = Session.latest(config.dataDir);
    session = latest
      ? Session.load(config.dataDir, latest)
      : Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
  } else {
    session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
  }

  console.log(`${cyan("◆")} ${bold("harness")} ${dim("·")} ${provider.name}/${provider.model}`);
  console.log(dim(`  cwd ${config.cwd}`));
  console.log(dim(`  session ${session.id} · :new :resume <id> :model <id> :system :context :exit`));

  let lastUsage: Usage | undefined;
  const spinner = makeSpinner();

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const ask = (q: string) => new Promise<string>((res) => rl.question(q, res));

  let ctrl: AbortController | null = null;
  rl.on("SIGINT", () => {
    if (ctrl) {
      ctrl.abort();
      process.stdout.write(dim("\n[turn aborted]\n"));
    } else {
      rl.close();
      process.exit(0);
    }
  });

  for (;;) {
    const input = (await ask(cyan("\n› "))).trim();
    if (!input) continue;

    if (input.startsWith(":")) {
      const [cmd, ...rest] = input.slice(1).split(/\s+/);
      switch (cmd) {
        case "exit": case "q": rl.close(); process.exit(0);
        case "new":
          session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
          console.log(dim(`new session ${session.id}`));
          continue;
        case "resume":
          try {
            session = Session.load(config.dataDir, rest[0] ?? "");
            console.log(dim(`resumed ${session.id} (${session.history.length} msgs)`));
          } catch (e) { console.log(dim((e as Error).message)); }
          continue;
        case "model":
          if (rest[0]) { provider = { ...provider, model: rest[0] }; console.log(dim(`model → ${provider.model}`)); }
          else console.log(dim(`model: ${provider.model}`));
          continue;
        case "system": {
          const sys = buildSystemPrompt(config);
          console.log(dim(`—— system prompt (${sys.length} chars, ~${Math.round(sys.length / 4)} tokens) ——`));
          console.log(sys);
          continue;
        }
        case "context": {
          const counts: Record<string, number> = {};
          let chars = 0;
          for (const m of session.history) {
            counts[m.role] = (counts[m.role] ?? 0) + 1;
            chars += JSON.stringify(m).length;
          }
          const sysChars = buildSystemPrompt(config).length;
          const total = chars + sysChars;
          const pct = Math.round((total / config.contextBudgetChars) * 100);
          const roles = Object.entries(counts).map(([r, n]) => `${n} ${r}`).join(", ") || "empty";
          console.log(dim(`session ${session.id}`));
          console.log(dim(`messages: ${session.history.length} (${roles})`));
          console.log(dim(`context: ~${Math.round(total / 4)} tokens (${total} chars incl. system) · ${pct}% of trim budget (${config.contextBudgetChars} chars)`));
          if (lastUsage) {
            console.log(dim(`last turn (provider-reported): ${lastUsage.prompt} prompt${lastUsage.cached ? ` (${lastUsage.cached} cached)` : ""} · ${lastUsage.completion} completion`));
          }
          continue;
        }
        default:
          console.log(dim(`unknown command :${cmd}`));
          continue;
      }
    }

    ctrl = new AbortController();
    let mode: "none" | "reason" | "text" = "none";
    const md = new MdRenderer();
    const active = new Map<string, string>(); // running tools: id → label
    const totals = { prompt: 0, cached: 0, completion: 0, steps: 0 };
    const t0 = performance.now();

    const spinnerLabel = () =>
      active.size === 0 ? "thinking" :
      active.size === 1 ? [...active.values()][0]! :
      `${active.size} tools running`;

    const endSegment = () => {
      if (mode === "text") {
        const rest = md.flush();
        if (rest) process.stdout.write(rest);
      }
      if (mode !== "none") process.stdout.write("\n");
      mode = "none";
    };

    const emit = (e: AgentEvent) => {
      spinner.stop();
      switch (e.type) {
        case "reasoning": {
          if (mode === "text") endSegment();
          let d = e.delta;
          if (mode !== "reason") {
            d = d.replace(/^\s+/, "");
            if (!d) break; // don't print a dangling ✻ for whitespace-only reasoning
            process.stdout.write(dim("✻ "));
            mode = "reason";
          }
          process.stdout.write(dim(d));
          break;
        }
        case "text":
          if (mode === "reason") process.stdout.write("\n\n");
          mode = "text";
          process.stdout.write(md.push(e.delta));
          break;
        case "tool_start": {
          endSegment();
          active.set(e.id, toolLabel(e.name, e.args));
          const diff = renderDiff(e.name, e.args);
          if (diff) process.stdout.write(diff);
          spinner.start(spinnerLabel());
          break;
        }
        case "tool_end": {
          const label = active.get(e.id) ?? e.name;
          active.delete(e.id);
          const mark = e.error ? red("✗") : green("✓");
          const summary = toolSummary(e.name, e.output, e.error);
          process.stdout.write(`${mark} ${label}${dim(` · ${summary} · ${e.ms}ms`)}\n`);
          spinner.start(spinnerLabel());
          break;
        }
        case "turn_end":
          endSegment();
          totals.steps++;
          if (e.usage) {
            lastUsage = e.usage;
            totals.prompt += e.usage.prompt;
            totals.cached += e.usage.cached ?? 0;
            totals.completion += e.usage.completion;
          }
          break;
        case "error":
          process.stdout.write(`\n${dim("error:")} ${e.message}\n`);
          break;
      }
    };

    spinner.start("thinking");
    try {
      await runTurn(config, provider, session, input, emit, ctrl.signal);
    } catch {
      // error already emitted
    } finally {
      spinner.stop();
      ctrl = null;
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      const cached = totals.cached ? ` (${totals.cached} cached)` : "";
      process.stdout.write(dim(`\n${totals.steps} step${totals.steps === 1 ? "" : "s"} · ↑${totals.prompt}${cached} ↓${totals.completion} · ${secs}s\n`));
    }
  }
}
