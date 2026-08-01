// Interactive REPL: slash-command menu, streaming markdown render, ctrl+o
// detail toggle (with retroactive reveal), ctrl+c aborts the turn.
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

// Accent bar marking agent-output lines (UI style C).
const BAR = "\x1b[2;36m▌\x1b[0m ";
/** Prefix every complete line in a streamed chunk with the accent bar. */
const barify = (chunk: string): string => {
  const parts = chunk.split("\n");
  return parts
    .map((l, i) => (i === parts.length - 1 ? l : l ? BAR + l : l))
    .join("\n");
};

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
  let lastPlan: string | null = null; // most recent plan-mode output
  let planArmed = false;              // attach the plan to the next code-mode message
  const spinner = makeSpinner();
  const history: string[] = [];

  // Details hidden by ctrl+o are buffered (capped) and replayed on toggle-on.
  const hidden: string[] = [];
  let hiddenBytes = 0;
  let hiddenLastWasReason = false;
  const stash = (text: string, isReason: boolean) => {
    if (isReason && !hiddenLastWasReason) hidden.push("\n✻ ");
    hiddenLastWasReason = isReason;
    hidden.push(text);
    hiddenBytes += text.length;
    while (hiddenBytes > 60_000 && hidden.length) {
      hiddenBytes -= hidden[0]!.length;
      hidden.shift();
    }
  };
  // Rows the last reveal block occupies on screen (wrap-aware); 0 = nothing
  // erasable. Invalidated the moment anything else prints.
  let revealRows = 0;
  const rowsAdvanced = (s: string): number => {
    const cols = Math.max(process.stdout.columns ?? 80, 20);
    const parts = s.split("\n");
    let rows = 0;
    for (let i = 0; i < parts.length - 1; i++) {
      const len = parts[i]!.replace(/\x1b\[[0-9;]*m/g, "").length;
      rows += 1 + Math.max(0, Math.ceil(len / cols) - 1);
    }
    return rows;
  };
  const toggleVerbose = () => {
    verbose = !verbose;
    if (verbose) {
      if (hidden.length) {
        const block =
          `\n${dim("[details on — ctrl+o hides them again]")}\n` +
          dim(hidden.join("").replace(/^\n+/, "")) + "\n";
        process.stdout.write(block);
        revealRows = rowsAdvanced(block);
      } else {
        process.stdout.write(`\n${dim("[details on — reasoning & diffs shown]")}\n`);
        revealRows = 0;
      }
    } else {
      if (revealRows > 0) {
        process.stdout.write(`\x1b[${revealRows}A\r\x1b[J`); // erase the reveal in place
        revealRows = 0;
      } else {
        process.stdout.write(`\n${dim("[details off — tool lines only]")}\n`);
      }
    }
  };

  emitKeypressEvents(process.stdin);

  let ctrl: AbortController | null = null;
  const abortTurn = () => {
    if (ctrl && !ctrl.signal.aborted) {
      ctrl.abort();
      process.stdout.write(dim("\n[turn aborted]\n"));
    }
  };
  // During a turn (readLine not active) handle ctrl+o / ctrl+c ourselves.
  process.stdin.on("keypress", (_s, key) => {
    if (!key?.ctrl || !ctrl) return;
    if (key.name === "o") toggleVerbose();
    else if (key.name === "c") abortTurn();
  });

  const exit = (code = 0): never => {
    console.log(dim(`\nsession ${session.id} — resume with: ap --resume ${session.id}`));
    process.exit(code);
  };

  for (;;) {
    const promptLabel = `${dim(config.mode)} ${cyan("›")} `;
    process.stdout.write("\n"); // once — NOT inside the prompt, which re-renders per keypress
    const input = (await readLine({
      prompt: promptLabel,
      commands: COMMANDS,
      history,
      onCtrlO: toggleVerbose,
    }))?.trim();

    if (input === null || input === undefined) exit(0); // ctrl+c at empty prompt
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
            // "opencode-go/minimax-m3" on the opencode-go provider → bare id
            provider = { ...provider, model: modelId };
            console.log(dim(`model → ${provider.model}`));
          } else if (config.providers[pfx]) {
            try {
              provider = resolveProvider(config, { provider: pfx, model: modelId } as CliFlags);
              console.log(dim(`provider → ${pfx}, model → ${modelId}`));
            } catch (e) { console.log(dim((e as Error).message)); }
          } else {
            // unknown provider → try the models.dev catalog
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

    ctrl = new AbortController();
    let mode: "none" | "reason" | "text" = "none";
    let planLines = 0;          // gist budget for plan-mode answers
    let planTruncated = false;  // rest of the plan goes to the browser only
    const PLAN_GIST_LINES = 10;
    const md = new MdRenderer();
    const active = new Map<string, string>();
    const totals = { prompt: 0, cached: 0, completion: 0, steps: 0 };
    const t0 = performance.now();

    const spinnerLabel = () =>
      active.size === 0 ? "thinking" :
      active.size === 1 ? [...active.values()][0]! :
      `${active.size} tools running`;

    const endSegment = () => {
      if (mode === "text") {
        const rest = md.flush();
        if (rest && !planTruncated) process.stdout.write(BAR + rest); // partial last line starts fresh — needs the bar
      }
      if (mode !== "none") process.stdout.write("\n");
      mode = "none";
    };

    const emit = (e: AgentEvent) => {
      spinner.stop();
      revealRows = 0; // new output below a reveal — it can no longer be erased in place
      switch (e.type) {
        case "reasoning": {
          if (!verbose) { stash(e.delta, true); break; }
          if (mode === "text") endSegment();
          let d = e.delta;
          if (mode !== "reason") {
            d = d.replace(/^\s+/, "");
            if (!d) break;
            process.stdout.write(BAR + dim("✻ "));
            mode = "reason";
          }
          process.stdout.write(dim(d.replace(/\n/g, "\n▌ ")));
          break;
        }
        case "text": {
          if (mode === "reason") process.stdout.write("\n\n");
          mode = "text";
          const rendered = md.push(e.delta);
          if (config.mode === "plan") {
            if (planTruncated) { spinner.start("writing plan…"); break; }
            process.stdout.write(barify(rendered));
            planLines += (rendered.match(/\n/g) ?? []).length;
            if (planLines >= PLAN_GIST_LINES) {
              planTruncated = true;
              process.stdout.write(dim("▌ … full plan opens in the browser when done\n"));
              spinner.start("writing plan…");
            }
          } else {
            process.stdout.write(barify(rendered));
          }
          break;
        }
        case "tool_start": {
          endSegment();
          active.set(e.id, toolLabel(e.name, e.args));
          if (config.mode !== "plan") {
            const diff = renderDiff(e.name, e.args);
            if (diff) {
              if (verbose) process.stdout.write(barify(diff));
              else stash(diff, false);
            }
          }
          spinner.start(spinnerLabel());
          break;
        }
        case "tool_end": {
          const label = active.get(e.id) ?? e.name;
          active.delete(e.id);
          const mark = e.error ? red("✗") : green("✓");
          const summary = toolSummary(e.name, e.output, e.error);
          process.stdout.write(`${BAR}${mark} ${label}${dim(` · ${summary} · ${e.ms}ms`)}\n`);
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

    // Fresh turn: old hidden details no longer apply.
    hidden.length = 0; hiddenBytes = 0; hiddenLastWasReason = false; revealRows = 0;

    let userText = input;
    if (planArmed && lastPlan && config.mode === "code") {
      userText += `\n\n<approved_plan>\n${lastPlan}\n</approved_plan>\nIf this message asks to implement the plan above, follow it EXACTLY — every step in order, nothing added, nothing skipped, no improvisation. If a step turns out to be impossible, stop and report instead of deviating.`;
      planArmed = false;
      process.stdout.write(dim("[plan attached — following it exactly]\n"));
    }

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
      process.stdout.write(dim(`\n${totals.steps} step${totals.steps === 1 ? "" : "s"} · ↑${totals.prompt}${cached} ↓${totals.completion} · ${secs}s\n`));
    }

    if (config.mode === "plan" && finalText.trim()) {
      lastPlan = finalText;
      if (finalText.length > 150) {
        try {
          const { exportPlanHtml, openInBrowser } = await import("./planview.ts");
          const planPath = exportPlanHtml(finalText, provider.model, config.cwd, session.id);
          openInBrowser(planPath);
          process.stdout.write(dim(`plan → ${planPath} (opened in browser) · /code to implement it exactly\n`));
        } catch (e) {
          process.stdout.write(dim(`plan export failed: ${(e as Error).message}\n`));
        }
      }
    }
  }
}
