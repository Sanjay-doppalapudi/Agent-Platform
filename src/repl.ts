// Interactive REPL: slash-command menu, streaming markdown render, ctrl+o
// detail toggle (true collapse/expand via in-place re-render), ctrl+c aborts.
import { emitKeypressEvents } from "node:readline";
import { loadConfig, resolveProvider } from "./config.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { buildSystemPrompt } from "./prompt.ts";
import { readLine, type SlashCommand } from "./input.ts";
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

// Accent bar marking agent-output lines.
const BAR = "\x1b[2;36m▌\x1b[0m ";
const barify = (chunk: string): string => {
  const parts = chunk.split("\n");
  return parts
    .map((l, i) => (i === parts.length - 1 ? l : l ? BAR + l : l))
    .join("\n");
};

const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

/** Terminal rows above the cursor that a printed chunk occupies (wrap-aware). */
function rowsUp(s: string): number {
  const cols = Math.max(process.stdout.columns ?? 80, 20);
  const parts = s.split("\n");
  let rows = 0;
  for (let i = 0; i < parts.length - 1; i++) {
    const len = stripAnsi(parts[i]!).length;
    rows += 1 + Math.max(0, Math.ceil(len / cols) - 1);
  }
  const tail = stripAnsi(parts[parts.length - 1]!).length;
  rows += Math.floor(tail / cols);
  return rows;
}

const COMMANDS: SlashCommand[] = [
  { name: "/plan", desc: "read-only mode: explore & produce a plan" },
  { name: "/code", desc: "full mode: all tools (default)" },
  { name: "/model", desc: "switch model, or provider/model", hasArg: true },
  { name: "/models", desc: "search the models.dev catalog", hasArg: true },
  { name: "/new", desc: "start a fresh session" },
  { name: "/resume", desc: "resume a session by id", hasArg: true },
  { name: "/sessions", desc: "list recent sessions" },
  { name: "/system", desc: "show the system prompt" },
  { name: "/context", desc: "show context/token usage" },
  { name: "/exit", desc: "quit" },
];

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

// A plan longer than this gets truncated in the terminal AND exported to an
// interactive HTML page — one threshold drives both decisions.
const PLAN_GIST_LINES = 30;

/**
 * Renders AgentEvents to a string. Two instances run in parallel per turn —
 * one including details (reasoning + diffs), one without — so ctrl+o can
 * re-render the whole turn in either form at any moment.
 */
class TurnRenderer {
  private md = new MdRenderer();
  private mode: "none" | "reason" | "text" = "none";
  private planLines = 0;
  planTruncated = false;

  constructor(
    private details: boolean,
    private planMode: boolean,
  ) {}

  feed(e: AgentEvent, label?: string): string {
    switch (e.type) {
      case "reasoning": {
        if (!this.details) return "";
        let out = "";
        let d = e.delta;
        if (this.mode === "text") out += this.endSegment();
        if (this.mode !== "reason") {
          d = d.replace(/^\s+/, "");
          if (!d) return out;
          out += BAR + dim("✻ ");
          this.mode = "reason";
        }
        return out + dim(d.replace(/\n/g, "\n▌ "));
      }
      case "text": {
        let out = "";
        if (this.mode === "reason") out += "\n\n";
        this.mode = "text";
        const rendered = this.md.push(e.delta);
        if (this.planMode) {
          if (this.planTruncated) return out;
          out += barify(rendered);
          this.planLines += (rendered.match(/\n/g) ?? []).length;
          if (this.planLines >= PLAN_GIST_LINES) {
            this.planTruncated = true;
            out += dim("▌ … long plan — the full version opens in the browser when done\n");
          }
        } else {
          out += barify(rendered);
        }
        return out;
      }
      case "tool_start": {
        let out = this.endSegment();
        if (this.details && !this.planMode) {
          const diff = renderDiff(e.name, e.args);
          if (diff) out += barify(diff);
        }
        return out;
      }
      case "tool_end": {
        const mark = e.error ? red("✗") : green("✓");
        const summary = toolSummary(e.name, e.output, e.error);
        return `${BAR}${mark} ${label ?? e.name}${dim(` · ${summary} · ${e.ms}ms`)}\n`;
      }
      case "turn_end":
        return this.endSegment();
      case "error":
        return `\n${dim("error:")} ${e.message}\n`;
      default:
        return "";
    }
  }

  endSegment(): string {
    let out = "";
    if (this.mode === "text") {
      const rest = this.md.flush();
      if (rest && !this.planTruncated) out += BAR + rest;
    }
    if (this.mode !== "none") out += "\n";
    this.mode = "none";
    return out;
  }
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

  config.sessionId = session.id;
  console.log(`${cyan("◆")} ${bold("AP")} ${dim("·")} ${provider.name}/${provider.model}`);
  console.log(dim(`  cwd ${config.cwd}`));
  console.log(dim(`  session ${session.id} · type / for commands · ctrl+o details · ctrl+c abort`));

  let lastUsage: Usage | undefined;
  let verbose = true;
  let lastPlan: string | null = null;
  let planArmed = false;
  const spinner = makeSpinner();
  const history: string[] = [];

  // Turn buffers: what a full render looks like, what a compact render looks
  // like, and what is actually on screen right now. ctrl+o erases the on-
  // screen block (cursor-up + clear) and prints the other form.
  let fullBuf = "";
  let compactBuf = "";
  let printedBuf = "";

  let ctrl: AbortController | null = null;

  const toggleVerbose = (atPrompt: boolean) => {
    verbose = !verbose;
    spinner.stop();
    const target = verbose ? fullBuf : compactBuf;
    const up = rowsUp(printedBuf) + (atPrompt ? 1 : 0); // +1: blank line before prompt
    const termRows = process.stdout.rows ?? 30;

    if (!printedBuf && !target) {
      // nothing rendered this turn yet — just confirm the flip
      const s = `${dim(`[details ${verbose ? "on" : "off"}]`)}\n`;
      process.stdout.write(s);
      printedBuf += s;
      return;
    }
    if (up > termRows - 1) {
      // Taller than the viewport: in-place erase is impossible, so clear the
      // visible screen and redraw the turn in the target form (older output
      // stays in scrollback).
      process.stdout.write(`\x1b[2J\x1b[H${target}`);
      printedBuf = target;
      if (atPrompt) process.stdout.write("\n");
      return;
    }
    if (up > 0) process.stdout.write(`\x1b[${up}A`);
    process.stdout.write(`\r\x1b[J${target}`);
    printedBuf = target;
    if (fullBuf === compactBuf) {
      const n = `${dim("[no hidden details in this turn]")}\n`;
      process.stdout.write(n);
      printedBuf += n;
    }
    if (atPrompt) process.stdout.write("\n"); // restore the blank line before the prompt
  };

  emitKeypressEvents(process.stdin);
  const abortTurn = () => {
    if (ctrl && !ctrl.signal.aborted) {
      ctrl.abort();
      process.stdout.write(dim("\n[turn aborted]\n"));
    }
  };
  // During a turn (readLine not active) handle ctrl+o / ctrl+c ourselves.
  process.stdin.on("keypress", (_s, key) => {
    if (!key?.ctrl || !ctrl) return;
    if (key.name === "o") toggleVerbose(false);
    else if (key.name === "c") abortTurn();
  });

  const exit = (code = 0): never => {
    console.log(dim(`\nsession ${session.id} — resume with: ap --resume ${session.id}`));
    process.exit(code);
  };

  for (;;) {
    const promptLabel = `${dim(config.mode)} ${cyan("›")} `;
    process.stdout.write("\n");
    const input = (await readLine({
      prompt: promptLabel,
      commands: COMMANDS,
      history,
      onCtrlO: () => toggleVerbose(true),
    }))?.trim();

    if (input === null || input === undefined) exit(0);
    if (!input) continue;
    history.push(input);

    if (input.startsWith("/") || input.startsWith(":")) {
      const [cmd, ...rest] = input.slice(1).split(/\s+/);
      switch (cmd) {
        case "exit": case "q": case "quit": exit(0);
        case "new":
          session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
          config.sessionId = session.id;
          console.log(dim(`new session ${session.id}`));
          continue;
        case "resume":
          try {
            session = Session.load(config.dataDir, rest[0] ?? "");
            config.sessionId = session.id;
            console.log(dim(`resumed ${session.id} (${session.history.length} msgs)`));
          } catch (e) { console.log(dim((e as Error).message)); }
          continue;
        case "session": case "sessions": {
          for (const s of Session.list(config.dataDir)) {
            const mark = s.id === session.id ? cyan("▸") : " ";
            console.log(`${mark} ${s.id} ${dim(new Date(s.mtime).toLocaleString())}`);
          }
          continue;
        }
        case "model": {
          const arg = rest[0];
          if (!arg) { console.log(dim(`model: ${provider.name}/${provider.model}`)); continue; }
          const slash = arg.indexOf("/");
          if (slash === -1) {
            provider = { ...provider, model: arg };
            console.log(dim(`model → ${provider.model}`));
            continue;
          }
          const pfx = arg.slice(0, slash);
          const modelId = arg.slice(slash + 1);
          if (pfx === provider.name) {
            provider = { ...provider, model: modelId };
            console.log(dim(`model → ${provider.model}`));
          } else if (config.providers[pfx]) {
            try {
              provider = resolveProvider(config, { provider: pfx, model: modelId } as CliFlags);
              console.log(dim(`provider → ${pfx}, model → ${modelId}`));
            } catch (e) { console.log(dim((e as Error).message)); }
          } else {
            try {
              const { loadCatalog, providerBaseUrl, envKeyFor } = await import("./models.ts");
              const { getKey } = await import("./creds.ts");
              const catalog = await loadCatalog(config.dataDir);
              const cp = catalog[pfx];
              const baseUrl = cp && providerBaseUrl(cp);
              if (!cp || !baseUrl) {
                console.log(dim(`unknown provider "${pfx}" — try /models ${pfx}`));
                continue;
              }
              const key = envKeyFor(cp) ?? getKey(config.dataDir, pfx);
              if (!key) {
                console.log(dim(`no key for ${pfx} — run: ap auth ${pfx}  (env: ${cp.env?.join("/") ?? "none listed"})`));
                continue;
              }
              provider = { name: pfx, baseUrl, apiKey: key, model: modelId, cacheControl: false, headers: {} };
              console.log(dim(`provider → ${pfx} via models.dev (${baseUrl}), model → ${modelId}`));
            } catch (e) { console.log(dim((e as Error).message)); }
          }
          continue;
        }
        case "models": {
          try {
            const { loadCatalog, searchModels } = await import("./models.ts");
            const catalog = await loadCatalog(config.dataDir);
            const rows = searchModels(catalog, rest.join(" "));
            if (!rows.length) { console.log(dim("no matches")); continue; }
            for (const r of rows) {
              const cost = r.inCost != null ? `$${r.inCost}/$${r.outCost ?? "?"}` : "";
              const ctx = r.ctx ? `${Math.round(r.ctx / 1000)}k` : "";
              console.log(`  ${r.provider}/${r.model} ${dim([ctx, cost].filter(Boolean).join(" · "))}`);
            }
            console.log(dim(`switch with /model <provider>/<model>`));
          } catch (e) { console.log(dim((e as Error).message)); }
          continue;
        }
        case "mode": case "plan": case "code": {
          const target = cmd === "mode" ? rest[0] : cmd;
          if (target === "plan" || target === "code") {
            config.mode = target;
            let note = target === "plan" ? " (read-only tools)" : "";
            if (target === "code" && lastPlan) {
              planArmed = true;
              note = " — plan armed: your next message carries the plan and it will be followed exactly";
            }
            console.log(dim(`mode → ${target}${note}`));
          } else console.log(dim(`mode: ${config.mode} (use /plan or /code)`));
          continue;
        }
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
          console.log(dim(`session ${session.id} · mode ${config.mode}`));
          console.log(dim(`messages: ${session.history.length} (${roles})`));
          console.log(dim(`context: ~${Math.round(total / 4)} tokens (${total} chars incl. system) · ${pct}% of trim budget (${config.contextBudgetChars} chars)`));
          if (lastUsage) {
            console.log(dim(`last turn (provider-reported): ${lastUsage.prompt} prompt${lastUsage.cached ? ` (${lastUsage.cached} cached)` : ""} · ${lastUsage.completion} completion`));
          }
          continue;
        }
        default:
          console.log(dim(`unknown command ${input.split(/\s+/)[0]} — type / to see commands`));
          continue;
      }
    }

    let userText = input;
    if (planArmed && lastPlan && config.mode === "code") {
      userText += `\n\n<approved_plan>\n${lastPlan}\n</approved_plan>\nIf this message asks to implement the plan above, follow it EXACTLY — every step in order, nothing added, nothing skipped, no improvisation. If a step turns out to be impossible, stop and report instead of deviating.`;
      planArmed = false;
      process.stdout.write(dim("[plan attached — following it exactly]\n"));
    }

    ctrl = new AbortController();
    const planMode = config.mode === "plan";
    const fullR = new TurnRenderer(true, planMode);
    const compactR = new TurnRenderer(false, planMode);
    fullBuf = ""; compactBuf = ""; printedBuf = "";

    const active = new Map<string, string>();
    const totals = { prompt: 0, cached: 0, completion: 0, steps: 0 };
    const t0 = performance.now();

    const spinnerLabel = () =>
      active.size === 0 ? "thinking" :
      active.size === 1 ? [...active.values()][0]! :
      `${active.size} tools running`;

    const writeBoth = (s: string) => {
      fullBuf += s; compactBuf += s;
      process.stdout.write(s);
      printedBuf += s;
    };

    const emit = (e: AgentEvent) => {
      spinner.stop();
      let label: string | undefined;
      if (e.type === "tool_start") {
        label = toolLabel(e.name, e.args);
        active.set(e.id, label);
      } else if (e.type === "tool_end") {
        label = active.get(e.id) ?? e.name;
        active.delete(e.id);
      } else if (e.type === "turn_end") {
        totals.steps++;
        if (e.usage) {
          lastUsage = e.usage;
          totals.prompt += e.usage.prompt;
          totals.cached += e.usage.cached ?? 0;
          totals.completion += e.usage.completion;
        }
      }

      const fullChunk = fullR.feed(e, label);
      const compactChunk = compactR.feed(e, label);
      fullBuf += fullChunk;
      compactBuf += compactChunk;
      const live = verbose ? fullChunk : compactChunk;
      if (live) {
        process.stdout.write(live);
        printedBuf += live;
      }

      if (e.type === "tool_start" || e.type === "tool_end") spinner.start(spinnerLabel());
      else if (e.type === "text" && planMode && fullR.planTruncated) spinner.start("writing plan…");
    };

    spinner.start("thinking");
    if (process.stdin.isTTY) process.stdin.setRawMode(true);
    let finalText = "";
    try {
      finalText = await runTurn(config, provider, session, userText, emit, ctrl.signal);
    } catch {
      // error already emitted
    } finally {
      spinner.stop();
      ctrl = null;
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      const cached = totals.cached ? ` (${totals.cached} cached)` : "";
      writeBoth(dim(`\n${totals.steps} step${totals.steps === 1 ? "" : "s"} · ↑${totals.prompt}${cached} ↓${totals.completion} · ${secs}s\n`));
    }

    if (config.mode === "plan" && finalText.trim()) {
      lastPlan = finalText;
      // Export only when the plan outgrew the terminal, or the user asked
      // for a page — short plans just print inline.
      const wantsHtml = /\b(html|browser|page)\b/i.test(input);
      if (fullR.planTruncated || wantsHtml) {
        try {
          const { exportPlanHtml, openInBrowser } = await import("./planview.ts");
          const planPath = exportPlanHtml(finalText, provider.model, config.cwd, session.id);
          openInBrowser(planPath);
          writeBoth(dim(`plan → ${planPath} (opened in browser) · /code to implement it exactly\n`));
        } catch (e) {
          writeBoth(dim(`plan export failed: ${(e as Error).message}\n`));
        }
      } else {
        writeBoth(dim(`plan ready · /code to implement it exactly\n`));
      }
    }
  }
}
