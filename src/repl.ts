// Interactive REPL: slash-command menu, streaming markdown render, ctrl+o
// detail toggle (true collapse/expand via in-place re-render), ctrl+c aborts.
import { emitKeypressEvents } from "node:readline";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { loadConfig, providerForMode, resolveProvider } from "./config.ts";
import { drainSteerNotes, pendingSteerCount, pushSteer } from "./steer.ts";
import { initMcp } from "./mcp.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { Checkpoints } from "./checkpoint.ts";
import { allIgnores, readRoots, sandboxRoots } from "./tools/shared.ts";
import { getTool, type PermitFn } from "./tools/index.ts";
import { listSubagents } from "./tools/agent.ts";
import { buildSystemPrompt, clearPromptSnapshots } from "./prompt.ts";
import { readLine, type SlashCommand } from "./input.ts";
import { MdRenderer } from "./md.ts";
import { errorHint, renderDiff, toolLabel, toolSummary } from "./ui.ts";
import { Session } from "./session.ts";
import { clearLive, publishLive } from "./live.ts";
import { compactSession, listArchives, restoreContextNote } from "./compact.ts";
import { streamChat } from "./provider.ts";
import { currentTheme, EFFORT_LEVELS, fitLine, fitSegments, frameBottom, frameTop, frameWidth, paint, parseEffort, setTheme, themeNames, THEMES, visibleLen, type EffortLevel, type StatusSegment } from "./theme.ts";
import type { Usage } from "./stream.ts";
import type { CliFlags } from "./index.ts";

// Colors resolve through the active theme on every call, so /theme switches
// take effect immediately without touching any of the ~80 call sites.
const dim = (s: string) => paint(currentTheme().dim, s);
const cyan = (s: string) => paint(currentTheme().accent, s);
const bold = (s: string) => paint(currentTheme().text, s);
const green = (s: string) => paint(currentTheme().success, s);
const red = (s: string) => paint(currentTheme().error, s);
const yellow = (s: string) => paint(currentTheme().warn, s);
const edge = (s: string) => paint(currentTheme().border, s);

// Visual hierarchy without an accent bar: answer text renders flush-left in
// the default color; everything that is NOT the answer (tool lines, diffs,
// reasoning, warnings) is indented two spaces and color-coded.
const IND = "  ";
const indent2 = (chunk: string): string => {
  const parts = chunk.split("\n");
  return parts
    .map((l, i) => (i === parts.length - 1 ? l : l ? IND + l : l))
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

const BUILTIN_CMDS = new Set([
  "exit", "q", "quit", "new", "resume", "session", "sessions", "rename", "sandbox",
  "model", "models", "mode", "plan", "code", "system", "context", "agents", "effort",
  "thinking", "confirm", "theme", "agent", "rewind",
  "undo", "diff", "checkpoints", "restore", "worktree", "compact", "archives",
  "restore-context", "share", "ps", "commit",
  "tasks", "flow", "artifacts", "watch", "steer",
]);

const COMMANDS: SlashCommand[] = [
  { name: "/plan", desc: "read-only mode: explore & produce a plan" },
  { name: "/code", desc: "full mode: all tools (default)" },
  { name: "/model", desc: "pick provider → model (filterable, models.dev), or /model <provider>/<model>", hasArg: true },
  { name: "/effort", desc: "reasoning effort: low|medium|high|off (models.dev-aware)", hasArg: true },
  { name: "/thinking", desc: "show or hide reasoning: on|off", hasArg: true },
  { name: "/confirm", desc: "confirm every edit/write: edits on|off", hasArg: true },
  { name: "/new", desc: "start a fresh session" },
  { name: "/resume", desc: "resume a session by id", hasArg: true },
  { name: "/sessions", desc: "list | delete <id> | rename <id> <title>", hasArg: true },
  { name: "/rename", desc: "set a title on the current session", hasArg: true },
  { name: "/steer", desc: "queue coaching for the next turn (or ctrl+s mid-turn)", hasArg: true },
  { name: "/sandbox", desc: "show or toggle the write-sandbox", hasArg: true },
  { name: "/theme", desc: "list or switch color themes", hasArg: true },
  { name: "/agents", desc: "list subagents spawned this session" },
  { name: "/tasks", desc: "background tasks: list | kill <id>", hasArg: true },
  { name: "/flow", desc: "list | last | <name> [args…] — .ap/workflows/<name>.ts", hasArg: true },
  { name: "/artifacts", desc: "list generated artifacts" },
  { name: "/watch", desc: "interactive viewer: agents, tasks, flows (←/→ switch · Esc back · ctrl+g mid-turn)" },
  { name: "/ps", desc: "background processes: list | tail <pid> | kill <pid>", hasArg: true },
  { name: "/skills", desc: "list available SKILL.md packs | reload", hasArg: true },
  { name: "/mcp", desc: "list MCP servers | reload (rebuilds tools; cache miss)", hasArg: true },
  { name: "/undo", desc: "restore the previous checkpoint" },
  { name: "/diff", desc: "checkpoint N | git | <branch> (vs merge-base)", hasArg: true },
  { name: "/checkpoints", desc: "list workspace checkpoints" },
  { name: "/restore", desc: "restore a checkpoint by hash", hasArg: true },
  { name: "/worktree", desc: "new <slug> | list | back | merge <slug>", hasArg: true },
  { name: "/commit", desc: "commit [--staged] [--sign] [message] — never pushes", hasArg: true },
  { name: "/pr", desc: "create a GitHub PR via gh [--draft] [--base <branch>] [title]", hasArg: true },
  { name: "/spawn", desc: "detach a task in tmux (unix) or print the Windows fallback", hasArg: true },
  { name: "/tmux", desc: "list | layout | capture <session> — optional unix adapter", hasArg: true },
  { name: "/compact", desc: "summarize history into a fresh session" },
  { name: "/archives", desc: "list recent compaction archives" },
  { name: "/restore-context", desc: "inject a note from an archived session id", hasArg: true },
  { name: "/rewind", desc: "drop last N user turns from history (not files)", hasArg: true },
  { name: "/agent", desc: "apply named agent profile to this session | clear", hasArg: true },
  { name: "/share", desc: "export the transcript as one self-contained HTML file" },
  { name: "/system", desc: "show the system prompt" },
  { name: "/context", desc: "show context/token usage" },
  { name: "/exit", desc: "quit" },
];

/** `extra` is appended to every frame — live elapsed/token state answers
 * "is it stuck?" with zero extra renders (the interval is already firing). */
/**
 * Live status line. Redraws IN PLACE at the bottom of the output with
 * `\r … \x1b[K`, so it is always the last thing on screen while a turn runs —
 * the model, context, cost, running agents and the keys that work right now.
 *
 * Deliberately NOT a terminal scrolling region (DECSTBM). A reserved band at
 * the screen's bottom edge sounds better, but DECSTBM homes the cursor every
 * time it is set, so enabling one mid-turn always displaces the output — it
 * left a block of blank rows between the prompt and the first token. Hiding
 * that needs a full-screen clear at startup, which would throw away the
 * normal terminal flow AP is built around. One in-place line has none of
 * those failure modes and works on every terminal, including legacy conhost.
 */
function makeSpinner(extra?: () => string, status?: () => string) {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let timer: ReturnType<typeof setInterval> | null = null;
  let i = 0;
  let label = "thinking";
  const render = () => {
    const cols = Math.max(process.stdout.columns ?? 80, 20);
    const head = `${frames[i++ % frames.length]} ${label}${extra?.() ?? ""}`;
    const tail = status?.() ?? "";
    const line = tail ? `${dim(head)}${dim(" · ")}${tail}` : dim(head);
    process.stdout.write(`\r${fitLine(line, cols - 1)}\x1b[K`);
  };
  return {
    start(newLabel = "thinking") {
      label = newLabel;
      if (timer || !process.stdout.isTTY) return;
      timer = setInterval(render, 100);
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
 * one including details (reasoning + subagent lines), one without — so ctrl+o
 * can re-render the whole turn in either form at any moment. Diffs show in
 * BOTH forms: they are decisions, not narration — a file must never change
 * without the diff having been visible.
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
          out += IND + dim("✻ ");
          this.mode = "reason";
        }
        return out + dim(d.replace(/\n/g, `\n${IND}`));
      }
      case "text": {
        let out = "";
        if (this.mode === "reason") out += "\n\n";
        this.mode = "text";
        const rendered = this.md.push(e.delta);
        if (this.planMode) {
          if (this.planTruncated) return out;
          out += rendered;
          this.planLines += (rendered.match(/\n/g) ?? []).length;
          if (this.planLines >= PLAN_GIST_LINES) {
            this.planTruncated = true;
            out += dim("… long plan — the full version opens in the browser when done\n");
          }
        } else {
          out += rendered;
        }
        return out;
      }
      case "tool_start": {
        let out = this.endSegment();
        if (!this.planMode) {
          const diff = renderDiff(e.name, e.args);
          if (diff) out += indent2(diff);
        }
        return out;
      }
      case "tool_end": {
        // Tool lines are indented + color-coded (cyan action, dim metadata)
        // so they can never be confused with the flush-left answer text.
        const mark = e.error ? red("✗") : green("✓");
        const summary = toolSummary(e.name, e.output, e.error);
        let out = `${IND}${mark} ${cyan(label ?? e.name)}${dim(` · ${summary} · ${e.ms}ms`)}\n`;
        if (e.name === "todo" && !e.error) {
          // The checklist IS the progress display — always show it.
          for (const l of e.output.split("\n").slice(1)) out += `${IND}${dim(`  ${l}`)}\n`;
        }
        return out;
      }
      case "turn_end":
        return this.endSegment();
      case "subline":
        // Nested subagent progress — details channel only.
        return this.details ? `${this.endSegment()}${IND}${dim(`  ${e.text}`)}\n` : "";
      case "warn":
        return `${this.endSegment()}${IND}${yellow(`⚠ ${e.message}`)}\n`;
      case "error": {
        const hint = errorHint(e.message);
        return `\n${dim("error:")} ${e.message}${hint ? `\n${dim(hint)}` : ""}\n`;
      }
      default:
        return "";
    }
  }

  endSegment(): string {
    let out = "";
    if (this.mode === "text") {
      const rest = this.md.flush();
      if (rest && !this.planTruncated) out += rest;
    }
    if (this.mode !== "none") out += "\n";
    this.mode = "none";
    return out;
  }
}

export async function replMain(flags: CliFlags) {
  const config = loadConfig(flags);
  if (config.theme) setTheme(config.theme);
  let provider: ReturnType<typeof resolveProvider>;
  /** Snapshotted code-mode provider so /plan → /code restores it when
   *  codeModel is unset (planModel alone must not permanently steal the slot). */
  let codeProvider: ReturnType<typeof resolveProvider>;
  try {
    provider = resolveProvider(config, flags);
    codeProvider = provider;
    // Per-mode model defaults (planModel/codeModel) apply when the user did
    // not pass -m / --model for this invocation.
    if (!flags.model && !process.env.HARNESS_MODEL) {
      if (config.mode === "code" && config.codeModel) {
        provider = providerForMode(config, "code", provider);
        codeProvider = provider;
      } else if (config.mode === "plan") {
        provider = providerForMode(config, "plan", provider);
      }
    }
  } catch (e) {
    if (Object.keys(config.providers).length || flags.baseUrl) throw e;
    // First run, nothing configured — a guided path beats a raw error.
    console.log(`${cyan("◆")} ${bold("AP")} — no provider configured yet. Two-minute setup:

  ${bold("1.")} Create ${cyan("ap.config.json")} in this project (or ~/.ap/config.json):
       ${dim(`{
         "provider": "openrouter",
         "providers": {
           "openrouter": { "baseUrl": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4.5" }
         }
       }`)}
  ${bold("2.")} Store your API key:   ${cyan("ap auth openrouter")}
  ${bold("3.")} Run:                  ${cyan("ap")}

  ${dim("Any OpenAI-compatible endpoint also works with zero config:")}
       ${cyan("ap --base-url <url> --api-key <key> -m <model>")}`);
    process.exit(1);
  }

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
  // Slant logo (16 cols wide) — printed only when the terminal fits it, so
  // the banner is layout-safe at any terminal size.
  const LOGO = [
    "    ___    ____ ",
    "   /   |  / __ \\",
    "  / /| | / /_/ /",
    " / ___ |/ ____/ ",
    "/_/  |_/_/      ",
  ];
  if (!config.light && (process.stdout.columns ?? 80) >= 20 && process.stdout.isTTY) {
    for (const l of LOGO) console.log(cyan(l));
    console.log(dim(`  ${provider.name}/${provider.model}`));
  } else {
    console.log(`${cyan("◆")} ${bold("AP")} ${dim("·")} ${provider.name}/${provider.model}${config.light ? dim(" · light") : ""}`);
  }
  console.log(dim(`  cwd ${config.cwd}`));
  console.log(dim(`  session ${session.id} · type / for commands · ctrl+o details · ctrl+c abort`));

  let lastUsage: Usage | undefined;
  let verbose = true;
  let lastPlan: string | null = null;
  let planArmed = false;
  // Live turn state for the spinner: elapsed seconds + rough output tokens.
  let turnT0 = 0;
  let turnOut = 0;
  /** Active tools with start time — spinner ticks per-tool elapsed. */
  let activeTools = new Map<string, { label: string; t0: number }>();
  /** Mid-turn steer draft (ctrl+s to start, Enter to queue). */
  let steerDraft = "";
  let steering = false;
  /** True while the sandbox permit prompt owns the keyboard. */
  let permitBusy = false;
  const spinner = makeSpinner(
    () => {
      if (!turnT0) return "";
      const s = Math.floor((performance.now() - turnT0) / 1000);
      let toolBit = "";
      if (activeTools.size === 1) {
        const t = [...activeTools.values()][0]!;
        const ts = ((performance.now() - t.t0) / 1000).toFixed(1);
        toolBit = ` · ${ts}s`;
      }
      return `${s >= 1 ? ` · ${s}s` : ""}${toolBit}${turnOut ? ` · ~${Math.round(turnOut / 4)} tok` : ""}`;
    },
    // Live status rides on the spinner line: always the last row of output,
    // always current, no cursor tricks.
    () => {
      if (config.light) return "";
      const steerBit = steering
        ? yellow(`steer: ${steerDraft || "…"}`)
        : pendingSteerCount()
          ? dim(`steer×${pendingSteerCount()}`)
          : "";
      const hint = steering ? dim("enter queue · esc cancel") : dim("ctrl+s steer · ctrl+g watch");
      return [statusFor(false), steerBit, hint].filter(Boolean).join(dim(" · "));
    },
  );
  const history: string[] = [];
  // One-time contextual hints: teach a feature the first time it matters.
  const hinted = new Set<string>();

  // Bottom bar pinned for the duration of each turn. Two rows: the status
  // (model · ctx · cost · agents · flow) framed like the prompt box, plus a
  // hint row. Only in the full profile on a TTY — --light stays frozen.

  // Reasoning effort: sent as `reasoning_effort` when set (/effort, config).
  let effort: EffortLevel | undefined = config.reasoningEffort;

  // Session spend for the status line. Pricing resolves lazily from the
  // models.dev catalog AFTER the first turn (never on the startup path).
  const spend = { prompt: 0, cached: 0, completion: 0 };
  let pricing: import("./models.ts").Pricing | null = null;
  let pricedFor = "";
  /** Name of the workflow running right now (status + live snapshot). */
  let activeFlow: string | null = null;
  let lastFlowName: string | null = null;
  /** REPL-applied named agent profile (null = none). */
  let agentRoleBody: string | null = null;
  let savedAgent: {
    toolFilter: string[] | undefined;
    provider: ReturnType<typeof resolveProvider>;
  } | null = null;
  /** Last computed context %, reused by the cross-process live snapshot so it
   *  never recomputes the (non-trivial) history scan. */
  let lastCtxPct = 0;

  /** `muted` paints the low-emphasis segments. Inside the frame it is the
   *  border color, so the status reads as part of the border instead of a
   *  white band across the bottom edge; standalone it is plain dim.
   *
   *  Responsive: segments carry drop priorities and are shed under width
   *  pressure (cost first, then effort, then agents — model and ctx% never
   *  drop; the model falls back to its short name instead). A zoomed-in
   *  terminal gets a status that FITS rather than a truncated one. */
  const buildStatus = (inFrame = false, budget = Infinity): string => {
    const muted = inFrame ? edge : dim;
    const sep = muted(" · ");
    let chars = 0;
    try {
      chars = buildSystemPrompt(config).length;
      for (const m of session.history) chars += JSON.stringify(m).length;
    } catch {}
    const pct = Math.min(999, Math.round((chars / config.contextBudgetChars) * 100));
    lastCtxPct = pct;
    const make = (modelText: string): StatusSegment[] => {
      const parts: StatusSegment[] = [{ text: muted(modelText), prio: 0 }];
      if (effort) parts.push({ text: muted(`effort ${effort}`), prio: 3 });
      parts.push({ text: pct >= 85 ? red(`ctx ${pct}%`) : pct >= 60 ? yellow(`ctx ${pct}%`) : muted(`ctx ${pct}%`), prio: 0 });
      if (pricing && (spend.prompt || spend.completion)) {
        // Inline the formula (same as models.estimateUsd) so the status path
        // never imports models.ts — catalog loading stays off the startup path.
        const cached = spend.cached;
        const usd =
          ((spend.prompt - cached) * pricing.input +
            cached * (pricing.cacheRead ?? pricing.input) +
            spend.completion * pricing.output) / 1e6;
        if (usd > 0) {
          const hit = spend.prompt > 0 && cached > 0
            ? `${Math.min(100, Math.round((cached / spend.prompt) * 100))}%`
            : "";
          const money = usd < 0.01 ? `~$${usd.toFixed(4)}` : usd < 1 ? `~$${usd.toFixed(3)}` : `~$${usd.toFixed(2)}`;
          parts.push({ text: muted(hit ? `${money} · cache ${hit}` : money), prio: 4 });
        }
      }
      const subs = listSubagents();
      if (subs.length) {
        const running = subs.filter((s) => s.status === "running").length;
        const dots = subs.slice(-8).map((s) =>
          s.status === "running" ? yellow("●") : s.status === "done" ? green("●") : red("●")).join("");
        // Running count first: "how many are working right now" is the live
        // question; the total is history.
        const label = running ? `◇ ${running}/${subs.length}` : `◇ ${subs.length}`;
        parts.push({ text: `${muted(label)} ${dots} ${muted("/agents")}`, prio: 2 });
      }
      if (activeFlow) parts.push({ text: `${cyan("◆ flow")} ${muted(activeFlow)}`, prio: 1 });
      return parts;
    };
    let s = fitSegments(make(`${provider.name}/${provider.model}`), budget, sep);
    if (visibleLen(s) > budget) {
      s = fitSegments(make(provider.model.split("/").pop() ?? provider.model), budget, sep);
    }
    return s;
  };
  /** Persist the chosen provider/model to <dataDir>/config.json, exactly like
   *  /theme — the next `ap` starts on the model you last picked instead of the
   *  config default. Returns a dim suffix for the confirmation line. */
  const rememberModel = (providerName: string, model: string): string => {
    try {
      const p = join(config.dataDir, "config.json");
      const cur = existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : {};
      cur.provider = providerName;
      cur.providers = { ...(cur.providers ?? {}), [providerName]: { ...(cur.providers?.[providerName] ?? {}), model } };
      writeFileSync(p, JSON.stringify(cur, null, 2) + "\n");
      return dim(" · saved as default");
    } catch (e) {
      return dim(` · not saved (${(e as Error).message})`);
    }
  };

  // Re-evaluated per keypress render — memoized so typing costs nothing. The
  // budget keys the cache too, so a zoom/resize rebuilds instead of serving a
  // status fitted to the previous width.
  let statusCache = { at: 0, s: "", frame: false, budget: 0 };
  const statusFor = (inFrame = false): string => {
    const budget = inFrame ? frameWidth() - 6 : Math.max(16, (process.stdout.columns ?? 80) - 3);
    const now = performance.now();
    if (now - statusCache.at > 1000 || statusCache.frame !== inFrame || statusCache.budget !== budget) {
      statusCache = { at: now, s: buildStatus(inFrame, budget), frame: inFrame, budget };
    }
    return statusCache.s;
  };

  /** Publish this process's state so a SEPARATE `ap watch` can render it —
   *  subagents/tasks/flows live in this process's memory and are otherwise
   *  invisible from another terminal. Throttled and best-effort inside. */
  // Background shell processes come from a JSONL file; re-reading it on every
  // tool event would be wasteful, so memoize for a couple of seconds.
  let procCache: { at: number; rows: import("./live.ts").LiveProc[] } = { at: 0, rows: [] };
  const bgProcs = (): import("./live.ts").LiveProc[] => {
    const now = Date.now();
    if (now - procCache.at < 2000) return procCache.rows;
    let rows: import("./live.ts").LiveProc[] = [];
    try {
      const { listBackground } = require("./bg.ts") as typeof import("./bg.ts");
      rows = listBackground(config).slice(0, 8)
        .map((p) => ({ pid: p.pid, cmd: p.cmd, log: p.log, alive: p.alive, bytes: p.bytes }));
    } catch {}
    procCache = { at: now, rows };
    return rows;
  };

  const publishStatus = (busy: boolean, force = false) => {
    if (config.light) return;
    try {
      const usd = pricing && (spend.prompt || spend.completion)
        ? ((spend.prompt - spend.cached) * pricing.input +
            spend.cached * (pricing.cacheRead ?? pricing.input) +
            spend.completion * pricing.output) / 1e6
        : undefined;
      publishLive(config.dataDir, {
        cwd: config.cwd,
        session: session.id,
        model: `${provider.name}/${provider.model}`,
        busy,
        ctxPct: lastCtxPct,
        usd,
        flow: activeFlow ?? undefined,
        agents: listSubagents().slice(-12).map((s) => ({
          id: s.id, label: s.task, status: s.status, steps: s.steps,
          startedAt: s.startedAt, background: s.background,
          fullTask: s.fullTask, result: s.result, live: s.live,
        })),
        procs: bgProcs(),
      }, force);
    } catch {}
  };

  /** One keypress from `keys` (Enter/Esc/ctrl+c → the last key = the safe no). */
  const readKey = (keys: string[]): Promise<string> =>
    new Promise((res) => {
      const no = keys[keys.length - 1]!;
      const onKey = (_s: string, key: any) => {
        if (!key) return;
        const done = (v: string) => { process.stdin.removeListener("keypress", onKey); res(v); };
        if (key.ctrl && key.name === "c") return done(no);
        if (key.ctrl) return;
        if (key.name === "return" || key.name === "enter" || key.name === "escape") return done(no);
        if (keys.includes(key.name)) return done(key.name);
      };
      process.stdin.on("keypress", onKey);
    });

  // Summarize the session into a fresh one (manual /compact + auto-compact).
  const compactNow = async (reason: "manual" | "auto" = "manual"): Promise<boolean> => {
    spinner.start("compacting…");
    try {
      const result = await compactSession({
        session,
        config,
        provider,
        checkpointId: cpSessionId,
        reason,
        idleTimeoutMs: config.streamIdleSeconds * 1000,
      });
      spinner.stop();
      // NB: cp is deliberately NOT rebound — rebinding created an empty repo
      // and silently orphaned every checkpoint made before the compaction.
      session = result.session;
      config.sessionId = session.id;
      sessionAllows = new Set();
      spend.prompt = spend.cached = spend.completion = 0;
      const mem = result.memoriesWritten ? ` · ${result.memoriesWritten} memor${result.memoriesWritten === 1 ? "y" : "ies"} saved` : "";
      console.log(dim(`compacted ${result.oldId} → new session ${session.id} (${result.summary.length} chars of summary) · checkpoint history preserved (/undo still works)${mem}`));
      return true;
    } catch (e) {
      spinner.stop();
      console.log(dim(`compact failed: ${(e as Error).message}`));
      return false;
    }
  };

  // Sandbox permission state: session-scoped "always allow" keys and a
  // promise chain so parallel tools never show two prompts at once.
  let sessionAllows = new Set<string>();
  let permitChain: Promise<unknown> = Promise.resolve();

  // The checkpoint repo is keyed by its OWN id, not the session id: /compact
  // starts a fresh session but must keep the existing undo trail reachable.
  // Resuming reads it back from the session meta so it survives a restart.
  let cpSessionId = session.meta?.checkpointId ?? session.id;
  let cp = new Checkpoints(config, cpSessionId);
  const originalCwd = config.cwd;
  if (session.recovered) {
    console.log(yellow(`⚠ session ${session.id} had a damaged line (crash mid-write) — it was skipped; the rest loaded fine`));
  }

  // MCP servers connect in the background while the user types; the promise
  // is awaited before the first turn so the tool list is complete + frozen.
  const mcpReady = initMcp(config, (m) => console.log(yellow(`⚠ ${m}`)));

  // Custom slash commands: .ap/commands/<name>.md in the repo or data dir.
  const customCommands = new Map<string, { file: string; desc: string }>();
  if (!config.light) {
    for (const dir of [join(config.cwd, ".ap", "commands"), join(config.dataDir, "commands")]) {
      try {
        for (const f of readdirSync(dir)) {
          if (!f.endsWith(".md")) continue;
          const name = basename(f, ".md").toLowerCase();
          if (customCommands.has(name)) continue;
          let desc = "custom command";
          try { desc = (readFileSync(join(dir, f), "utf8").split("\n")[0] ?? "").replace(/^#+\s*/, "").slice(0, 40) || desc; } catch {}
          customCommands.set(name, { file: join(dir, f), desc });
        }
      } catch {}
    }
  }
  let flowMenu: SlashCommand[] = [];
  if (!config.light) {
    try {
      const { listFlows } = await import("./flow.ts");
      flowMenu = listFlows(config).map((f) => ({
        name: `/flow ${f.name}`,
        desc: "workflow",
        hasArg: true,
      }));
    } catch {}
  }
  const menuCommands = [
    ...COMMANDS,
    ...flowMenu,
    ...[...customCommands.entries()].map(([n, c]) => ({ name: `/${n}`, desc: c.desc, hasArg: true })),
  ];

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
  /** Open the interactive viewer in this terminal (alternate screen), then
   *  restore whatever was on screen. Works mid-turn too — the turn keeps
   *  running underneath, which is the point. */
  let watching = false;
  /** Turn output produced while the viewer is up, replayed on return. */
  let held = "";
  const openWatch = async () => {
    if (watching || config.light || !process.stdout.isTTY) return;
    const wasBusy = !!ctrl;
    try {
      spinner.stop(); // its 80ms timer would scribble over the viewer
      publishStatus(!!ctrl, true); // make our own row fresh before drawing
      watching = true; // from here on, turn output is held, not written
      const { runWatch } = await import("./watch.ts");
      await runWatch({ dataDir: config.dataDir });
    } catch (e) {
      console.log(dim(`watch failed: ${(e as Error).message}`));
    } finally {
      watching = false;
      // Replay what the turn produced while we were away — the alternate
      // screen discarded nothing because nothing was written to it.
      if (held) { process.stdout.write(held); held = ""; }
      if (ctrl) spinner.start("thinking"); // the turn is still running
    }
  };

  // During a turn (readLine not active): ctrl+g/o/c, plus ctrl+s steer queue.
  process.stdin.on("keypress", (ch, key) => {
    if (!ctrl || permitBusy) return;
    if (key?.ctrl) {
      // ctrl+G opens the viewer. NOT ctrl+W: Windows Terminal binds that to
      // close-pane by default, so the keypress never reaches us — it is kept
      // only as an alias for terminals that do deliver it.
      if (key.name === "g" || key.name === "w") { void openWatch(); return; }
      if (key.name === "o") { toggleVerbose(false); return; }
      if (key.name === "c") { abortTurn(); return; }
      if (key.name === "s") {
        steering = !steering;
        if (!steering) steerDraft = "";
        else if (!hinted.has("steer")) {
          hinted.add("steer");
          process.stdout.write(`\n${dim("  steer mode — type a note, Enter queues it for the next turn, Esc cancels")}\n`);
        }
        return;
      }
      return;
    }
    if (!steering) return;
    if (key?.name === "escape") { steering = false; steerDraft = ""; return; }
    if (key?.name === "return" || key?.name === "enter") {
      if (pushSteer(steerDraft)) {
        process.stdout.write(`\n${dim(`  queued steer (${pendingSteerCount()}): ${steerDraft.replace(/\s+/g, " ").slice(0, 80)}`)}\n`);
      }
      steerDraft = "";
      steering = false;
      return;
    }
    if (key?.name === "backspace") {
      steerDraft = steerDraft.slice(0, -1);
      return;
    }
    // Printable character (raw mode delivers it as `ch`).
    if (typeof ch === "string" && ch.length === 1 && ch >= " ") {
      if (steerDraft.length < 500) steerDraft += ch;
    }
  });

  const exit = (code = 0): never => {
    clearLive(config.dataDir); // stop advertising this process to `ap watch`
    console.log(dim(`\nsession ${session.id} — resume with: ap --resume ${session.id}`));
    process.exit(code);
  };
  process.once("exit", () => clearLive(config.dataDir));

  for (;;) {
    // Framed input: top edge (with the mode as its label), a left edge on the
    // input line itself, and the status row doubling as the bottom edge. Only
    // the input line carries a left edge, so wrapped text never breaks the box.
    const framed = !config.light && process.stdout.isTTY;
    const promptLabel = framed
      ? `${edge("│")} ${cyan("›")} `
      : `${dim(config.mode)} ${cyan("›")} `;
    process.stdout.write("\n");
    let input = (await readLine({
      prompt: promptLabel,
      commands: menuCommands,
      history,
      files: config.light ? undefined : () => {
        // Workspace file list for @-completion — pruned by ripgrep, capped.
        const rgArgs = ["--files", "--hidden"];
        for (const ig of allIgnores(config)) rgArgs.push("-g", `!**/${ig}/**`, "-g", `!${ig}/**`);
        const p = Bun.spawnSync(["rg", ...rgArgs], { cwd: config.cwd, stdout: "pipe", stderr: "ignore" });
        return (p.stdout?.toString() ?? "")
          .split("\n")
          .filter(Boolean)
          .slice(0, 800)
          .map((f) => f.replace(/\\/g, "/"));
      },
      onCtrlO: () => toggleVerbose(true),
      // readLine owns the frame: it redraws the top edge on every render, so
      // the box survives resizes, wrapped input, and ctrl+o replays. The mode
      // label carries its own color (plan = warn, code = accent) via painted
      // pieces — never a wrapper around the whole line (ANSI nesting rule).
      frameTop: framed
        ? () => frameTop(
            frameWidth(),
            config.mode === "plan" ? "▲ plan" : "◆ code",
            edge,
            config.mode === "plan" ? yellow : cyan,
          )
        : undefined,
      status: config.light ? undefined : () => (framed ? frameBottom(frameWidth(), statusFor(true), edge) : `  ${statusFor()}`),
    }));
    // Submitting erases the live status row (it is the bottom edge), so close
    // the box explicitly — otherwise the transcript keeps an open-ended frame.
    if (framed) console.log(edge(frameBottom(frameWidth())));
    input = input === null ? null : input.trim();

    if (input === null || input === undefined) exit(0);
    if (!input) continue;
    history.push(input);

    if (input.startsWith("/") || input.startsWith(":")) {
      const [cmd, ...rest] = input.slice(1).split(/\s+/);
      // Custom command? (builtins always win) — expand template and run as a turn.
      const custom = customCommands.get((cmd ?? "").toLowerCase());
      if (custom && !BUILTIN_CMDS.has((cmd ?? "").toLowerCase())) {
        try {
          input = readFileSync(custom.file, "utf8").replace(/\$(ARGUMENTS|ARGS)\b/g, rest.join(" ")).trim();
          process.stdout.write(dim(`[/${cmd} → ${basename(custom.file)}]\n`));
        } catch (e) {
          console.log(dim(`failed to read ${custom.file}: ${(e as Error).message}`));
          continue;
        }
      } else {
      switch (cmd) {
        case "exit": case "q": case "quit": exit(0);
        case "new":
          session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
          config.sessionId = session.id;
          sessionAllows = new Set();
          spend.prompt = spend.cached = spend.completion = 0;
          cpSessionId = session.id;
          cp = new Checkpoints(config, cpSessionId);
          console.log(dim(`new session ${session.id}`));
          continue;
        case "resume":
          try {
            session = Session.load(config.dataDir, rest[0] ?? "");
            config.sessionId = session.id;
            sessionAllows = new Set();
            spend.prompt = spend.cached = spend.completion = 0;
            cpSessionId = session.meta?.checkpointId ?? session.id;
            cp = new Checkpoints(config, cpSessionId);
            console.log(dim(`resumed ${session.id} (${session.history.length} msgs)`));
          } catch (e) { console.log(dim((e as Error).message)); }
          continue;
        case "sandbox": {
          const arg = rest[0];
          if (arg === "on") config.sandbox = "workspace";
          else if (arg === "off") config.sandbox = "off";
          if (arg === "on" || arg === "off") {
            console.log(dim(`sandbox → ${config.sandbox} (system prompt changed — one-time cache re-prime)`));
            continue;
          }
          console.log(dim(`sandbox: ${config.sandbox} · bashGuard: ${config.bashGuard}`));
          console.log(dim(`writable roots: ${sandboxRoots(config).join(" · ")}`));
          console.log(dim(`readable roots: ${readRoots(config).join(" · ")} (elsewhere prompts; AP sessions/credentials always blocked)`));
          if (sessionAllows.size) console.log(dim(`session allows: ${[...sessionAllows].join(", ")}`));
          console.log(dim(`/sandbox on|off to toggle`));
          continue;
        }
        case "rename": {
          const title = rest.join(" ").trim();
          if (!title) {
            console.log(dim(session.meta?.title
              ? `title: ${session.meta.title} — /rename <title> to change, /sessions rename ${session.id} "" to clear`
              : `no title — /rename <title>`));
            continue;
          }
          try {
            session = Session.rename(config.dataDir, session.id, title);
            console.log(dim(`renamed → ${session.meta?.title}`));
          } catch (e) { console.log(dim((e as Error).message)); }
          continue;
        }
        case "session": case "sessions": {
          const sub = rest[0]?.toLowerCase();
          if (sub === "delete" || sub === "rm") {
            const id = rest[1];
            if (!id) { console.log(dim("usage: /sessions delete <id>")); continue; }
            if (id === session.id) { console.log(dim("can't delete the active session — /new or /resume another first")); continue; }
            try {
              Session.load(config.dataDir, id); // confirm exists
              Session.delete(config.dataDir, id);
              console.log(dim(`deleted ${id}`));
            } catch (e) { console.log(dim((e as Error).message)); }
            continue;
          }
          if (sub === "rename") {
            const id = rest[1];
            const title = rest.slice(2).join(" ");
            if (!id) { console.log(dim("usage: /sessions rename <id> <title>")); continue; }
            try {
              const s = Session.rename(config.dataDir, id, title);
              if (id === session.id) session = s;
              console.log(dim(s.meta?.title ? `renamed ${id} → ${s.meta.title}` : `cleared title on ${id}`));
            } catch (e) { console.log(dim((e as Error).message)); }
            continue;
          }
          for (const s of Session.list(config.dataDir)) {
            const mark = s.id === session.id ? cyan("▸") : " ";
            let title = "";
            try { title = Session.load(config.dataDir, s.id).meta?.title ?? ""; } catch {}
            console.log(`${mark} ${s.id}${title ? ` ${cyan(title)}` : ""} ${dim(new Date(s.mtime).toLocaleString())}`);
          }
          if (!sub) console.log(dim("/sessions delete <id> · /sessions rename <id> <title> · /rename <title>"));
          continue;
        }
        case "steer": {
          const text = rest.join(" ").trim();
          if (!text) {
            const n = pendingSteerCount();
            console.log(dim(n
              ? `${n} steer note(s) queued for the next turn — /steer <text> to add more`
              : "usage: /steer <text> — or ctrl+s mid-turn to queue coaching without aborting"));
            continue;
          }
          pushSteer(text);
          console.log(dim(`queued steer (${pendingSteerCount()}): ${text.slice(0, 100)}`));
          continue;
        }
        case "model": {
          const arg = rest[0];
          if (!arg) {
            // Interactive: providers from models.dev + config, key prompt if
            // missing, then that provider's models. Full profile + TTY only —
            // --light keeps the old one-line answer (frozen surface).
            if (config.light || !process.stdout.isTTY) {
              console.log(dim(`model: ${provider.name}/${provider.model}${config.light ? "" : " — /model <provider>/<model> to switch"}`));
              continue;
            }
            try {
              const { pickModelInteractive } = await import("./picker.ts");
              const next = await pickModelInteractive(config, provider);
              if (next) {
                provider = next;
                if (config.mode === "code") codeProvider = provider;
                pricedFor = "";
                console.log(dim(`provider → ${next.name} (${next.baseUrl}), model → ${next.model}${rememberModel(next.name, next.model)}`));
              } else {
                console.log(dim(`model: ${provider.name}/${provider.model} (unchanged)`));
              }
            } catch (e) { console.log(dim((e as Error).message)); }
            continue;
          }
          const slash = arg.indexOf("/");
          if (slash === -1) {
            provider = { ...provider, model: arg };
            if (config.mode === "code") codeProvider = provider;
            pricedFor = "";
            console.log(dim(`model → ${provider.model}${rememberModel(provider.name, provider.model)}`));
            continue;
          }
          const pfx = arg.slice(0, slash);
          const modelId = arg.slice(slash + 1);
          if (pfx === provider.name) {
            provider = { ...provider, model: modelId };
            if (config.mode === "code") codeProvider = provider;
            pricedFor = "";
            console.log(dim(`model → ${provider.model}${rememberModel(provider.name, provider.model)}`));
          } else if (config.providers[pfx]) {
            try {
              provider = resolveProvider(config, { provider: pfx, model: modelId } as CliFlags);
              if (config.mode === "code") codeProvider = provider;
              pricedFor = "";
              console.log(dim(`provider → ${pfx}, model → ${modelId}${rememberModel(pfx, modelId)}`));
            } catch (e) { console.log(dim((e as Error).message)); }
          } else {
            try {
              const { resolveCatalogProvider } = await import("./models.ts");
              provider = await resolveCatalogProvider(config, pfx, modelId);
              if (config.mode === "code") codeProvider = provider;
              pricedFor = "";
              console.log(dim(`provider → ${pfx} via models.dev/config (${provider.baseUrl}), model → ${modelId}${rememberModel(pfx, modelId)}`));
            } catch (e) { console.log(dim((e as Error).message)); }
          }
          continue;
        }
        case "theme": {
          const arg = rest[0]?.toLowerCase();
          if (!arg) {
            for (const n of themeNames()) {
              const t = THEMES[n]!;
              const swatch = `${paint(t.accent, "●")}${paint(t.success, "●")}${paint(t.warn, "●")}${paint(t.error, "●")}${paint(t.border, "●")}`;
              console.log(`${swatch} ${n === currentTheme().name ? cyan(`${n} ✓`) : n} ${dim(t.desc)}`);
            }
            console.log(dim(`/theme <name> to switch (saved to ${join(config.dataDir, "config.json")})`));
            continue;
          }
          if (!setTheme(arg)) {
            console.log(dim(`unknown theme "${arg}" — available: ${themeNames().join(", ")}`));
            continue;
          }
          // Persist the choice: a UI preference the user explicitly asked for.
          let saved = "";
          try {
            const p = join(config.dataDir, "config.json");
            const cur = existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : {};
            cur.theme = currentTheme().name;
            writeFileSync(p, JSON.stringify(cur, null, 2) + "\n");
            saved = ` · saved to ${p}`;
          } catch (e) { saved = ` · not saved (${(e as Error).message})`; }
          console.log(`${cyan(`theme → ${currentTheme().name}`)}${dim(saved)}`);
          continue;
        }
        case "thinking": {
          const arg = rest[0]?.toLowerCase();
          if (arg === "on" || arg === "off") {
            config.showReasoning = arg;
            console.log(dim(`thinking → ${arg}`));
          } else {
            console.log(dim(`thinking: ${config.showReasoning ?? "on"} — /thinking on|off`));
          }
          continue;
        }
        case "confirm": {
          const a0 = rest[0]?.toLowerCase();
          const a1 = rest[1]?.toLowerCase();
          if (a0 === "edits" && (a1 === "on" || a1 === "off")) {
            config.confirmEdits = a1 === "on";
            console.log(dim(`confirm edits → ${a1}`));
          } else {
            console.log(dim(`confirm edits: ${config.confirmEdits ? "on" : "off"} — /confirm edits on|off`));
          }
          continue;
        }
        case "effort": {
          const arg = rest[0]?.toLowerCase();
          if (!arg) {
            console.log(dim(`reasoning effort: ${effort ?? "provider default"} — /effort low | medium | high | off`));
            continue;
          }
          const level = parseEffort(arg);
          if (!level) {
            console.log(dim(`unknown effort "${arg}" — use ${EFFORT_LEVELS.join(" | ")} | off (aliases: max, min, med…)`));
            continue;
          }
          if (level === "off") {
            effort = undefined;
            console.log(dim("reasoning effort → provider default"));
            continue;
          }
          effort = level;
          let note = "";
          try {
            const { loadCatalog, modelReasoning } = await import("./models.ts");
            const sup = modelReasoning(await loadCatalog(config.dataDir), provider.name, provider.model);
            if (sup === false) note = " — models.dev: this model does NOT advertise reasoning; the provider may ignore or reject the parameter";
            else if (sup === true) note = " (model supports reasoning per models.dev)";
          } catch {}
          console.log(dim(`reasoning effort → ${arg}${note}`));
          continue;
        }
        case "models": {
          // Merged into /model: the picker IS the catalog search (type to
          // filter). Kept as a redirect so muscle memory doesn't send
          // "/models foo" to the model as prompt text.
          console.log(dim(`/models was merged into /model — it opens a filterable provider → model picker (ap models <query> still works from the shell)`));
          continue;
        }
        case "mode": case "plan": case "code": {
          const target = cmd === "mode" ? rest[0] : cmd;
          if (target === "plan" || target === "code") {
            if (config.mode === "code") codeProvider = provider;
            config.mode = target;
            const prev = provider;
            if (target === "plan") {
              provider = config.planModel
                ? providerForMode(config, "plan", provider)
                : provider;
            } else {
              provider = config.codeModel
                ? providerForMode(config, "code", codeProvider)
                : codeProvider;
            }
            let modelNote = "";
            if (prev.name !== provider.name || prev.model !== provider.model) {
              modelNote = ` · model ${provider.name}/${provider.model}`;
              pricedFor = ""; // refresh pricing after switch
            }
            let note = target === "plan" ? " (read-only tools)" : "";
            if (target === "code" && lastPlan) {
              planArmed = true;
              note = " — plan armed: your next message carries the plan and it will be followed exactly";
            }
            console.log(dim(`mode → ${target}${note}${modelNote}`));
          } else {
            const pm = config.planModel ? ` · planModel ${config.planModel}` : "";
            const cm = config.codeModel ? ` · codeModel ${config.codeModel}` : "";
            console.log(dim(`mode: ${config.mode}${pm}${cm} (use /plan or /code)`));
          }
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
          const title = session.meta?.title ? ` · ${session.meta.title}` : "";
          console.log(dim(`session ${session.id}${title} · mode ${config.mode} · ${provider.name}/${provider.model}`));
          console.log(dim(`messages: ${session.history.length} (${roles})`));
          console.log(dim(`context: ~${Math.round(total / 4)} tokens (${total} chars incl. system) · ${pct}% of trim budget (${config.contextBudgetChars} chars)`));
          if (lastUsage) {
            const hit = lastUsage.cached && lastUsage.prompt
              ? ` · cache ${Math.min(100, Math.round((lastUsage.cached / lastUsage.prompt) * 100))}%`
              : "";
            console.log(dim(`last turn: ↑${lastUsage.prompt}${lastUsage.cached ? ` (${lastUsage.cached} cached)` : ""}${hit} ↓${lastUsage.completion}`));
            if (pricing) {
              const { estimateUsd, formatUsd } = await import("./models.ts");
              const lastUsd = estimateUsd(pricing, {
                prompt: lastUsage.prompt,
                cached: lastUsage.cached,
                completion: lastUsage.completion,
              });
              console.log(dim(`cost: last ${formatUsd(lastUsd)} · session ${formatUsd(estimateUsd(pricing, spend))}`));
            }
          } else if (pricing && (spend.prompt || spend.completion)) {
            const { estimateUsd, formatUsd, cacheHitPct } = await import("./models.ts");
            const hit = cacheHitPct(spend.prompt, spend.cached);
            console.log(dim(`session spend: ${formatUsd(estimateUsd(pricing, spend))}${hit ? ` · cache ${hit}` : ""}`));
          }
          continue;
        }
        case "flow": {
          if (config.light) { console.log(dim("workflows are not available in --light")); continue; }
          const { runFlow, listFlows } = await import("./flow.ts");
          const name = rest[0];
          if (!name || name === "list") {
            const flows = listFlows(config);
            if (!flows.length) {
              console.log(dim("no workflows — add .ap/workflows/<name>.ts"));
              continue;
            }
            for (const f of flows) console.log(`${cyan(f.name)} ${dim(f.path)}`);
            if (lastFlowName) console.log(dim(`last: ${lastFlowName} — /flow last to rerun`));
            continue;
          }
          const runName = name === "last" ? lastFlowName : name;
          if (!runName) {
            console.log(dim(name === "last" ? "no previous flow this session" : "usage: /flow <name> | list | last"));
            continue;
          }
          try {
            activeFlow = runName;
            publishStatus(true, true);
            const result = await runFlow(config, runName, name === "last" ? [] : rest.slice(1), (line) => {
              console.log(dim(`  ${line}`));
              publishStatus(true);
            });
            lastFlowName = runName;
            if (result !== undefined) {
              console.log(typeof result === "string" ? result : JSON.stringify(result, null, 2));
            }
            console.log(dim(`flow ${runName} finished`));
          } catch (e) { console.log(red(`flow failed: ${(e as Error).message}`)); }
          finally { activeFlow = null; publishStatus(false, true); }
          continue;
        }
        case "watch": {
          await openWatch();
          continue;
        }
        case "artifacts": {
          if (config.light) { console.log(dim("artifacts are not available in --light")); continue; }
          try {
            const { readdirSync, statSync } = await import("node:fs");
            const dir = join(config.dataDir, "artifacts");
            const files = readdirSync(dir).filter((f) => f.endsWith(".html"))
              .map((f) => ({ f, m: statSync(join(dir, f)).mtimeMs }))
              .sort((a, b) => b.m - a.m).slice(0, 20);
            if (!files.length) { console.log(dim("no artifacts yet — the artifact tool writes them")); continue; }
            for (const { f, m } of files) {
              console.log(`  ${cyan(f)} ${dim(new Date(m).toLocaleString())}`);
            }
            console.log(dim(`open: ${join(dir, files[0]!.f)}`));
          } catch { console.log(dim("no artifacts yet — the artifact tool writes them")); }
          continue;
        }
        case "tasks": {
          const { listTasks, taskById } = await import("./tasks.ts");
          const sub = rest[0]?.toLowerCase();
          if (sub === "kill") {
            const t = taskById(Number(rest[1]));
            if (!t) { console.log(dim(`no task #${rest[1] ?? "?"} — /tasks to list`)); continue; }
            if (t.status !== "running") { console.log(dim(`task #${t.id} already ${t.status}`)); continue; }
            t.ctrl.abort();
            console.log(dim(`task #${t.id} kill requested`));
            continue;
          }
          const all = listTasks();
          if (!all.length) { console.log(dim("no background tasks this session — agent {background:true} starts one")); continue; }
          for (const t of all) {
            const secs = (((t.endedAt ?? Date.now()) - t.startedAt) / 1000).toFixed(1);
            const mark = t.status === "running" ? yellow("●") : t.status === "done" ? green("✓") : red("✗");
            console.log(`${mark} #${t.id} ${dim(`[${t.status}]`)} ${t.task} ${dim(`· ${t.steps} steps · ${secs}s`)}`);
            if (t.result && t.status !== "done") console.log(dim(`    ${t.result.split("\n")[0]?.slice(0, 100)}`));
          }
          console.log(dim(`/tasks kill <id> stops one · results fold into your next message`));
          continue;
        }
        case "agents": {
          const { discoverAgents } = await import("./agents.ts");
          const defs = discoverAgents(config);
          if (defs.length) {
            console.log(dim("defined agents (agent tool name:… / ap run --agent …):"));
            for (const a of defs) {
              console.log(`${cyan(a.name)} ${dim(`[${a.source}]`)}${a.model ? dim(` ${a.model}`) : ""} ${(a.description || "").slice(0, 70)}`);
            }
          } else {
            console.log(dim("no agent profiles — add .ap/agents/<name>.md (frontmatter: description/model/tools; body = role)"));
          }
          const subs = listSubagents();
          if (!subs.length) { console.log(dim("no subagents spawned this session")); continue; }
          for (const s of subs) {
            const secs = (((s.endedAt ?? Date.now()) - s.startedAt) / 1000).toFixed(1);
            const mark = s.status === "running" ? yellow("●") : s.status === "done" ? green("✓") : red("✗");
            console.log(`${mark} #${s.id} ${dim(`[${s.status}]`)} ${s.task} ${dim(`· ${s.steps} steps · ${secs}s`)}`);
          }
          continue;
        }
        case "ps": {
          const { listBackground, tailLog, killBackground, formatBytes } = await import("./bg.ts");
          const sub = rest[0];
          if (sub === "kill" && rest[1]) {
            const r = killBackground(config, Number(rest[1]));
            console.log(dim(r.message));
            continue;
          }
          const rows = listBackground(config);
          if (sub === "tail") {
            const pick = rest[1] ? rows.find((r) => r.pid === Number(rest[1])) : rows.find((r) => r.alive) ?? rows[0];
            if (!pick) { console.log(dim(rest[1] ? `no background process with pid ${rest[1]}` : "no background processes")); continue; }
            console.log(dim(`— ${pick.cmd} (pid ${pick.pid}) · ${pick.log} —`));
            console.log(tailLog(pick.log, Number(rest[2]) || 40));
            continue;
          }
          if (!rows.length) { console.log(dim("no background processes — start one with bash background:true")); continue; }
          for (const r of rows) {
            const mark = r.alive ? green("●") : dim("○");
            console.log(`${mark} ${cyan(String(r.pid))} ${r.cmd.slice(0, 60)} ${dim(`· ${r.alive ? "running" : "exited"} · ${formatBytes(r.bytes)} · ${new Date(r.at).toLocaleTimeString()}`)}`);
          }
          console.log(dim("/ps tail <pid> [lines] · /ps kill <pid>"));
          continue;
        }
        case "skills": {
          const { clearSkillCache, discoverSkills } = await import("./skills.ts");
          if (rest[0] === "reload") {
            clearSkillCache();
            clearPromptSnapshots(session.id);
            const skills = discoverSkills(config);
            console.log(dim(`skills reloaded (${skills.length}) — prompt prefix may miss cache until next turns settle`));
            continue;
          }
          const skills = discoverSkills(config);
          if (!skills.length) { console.log(dim("no skills — install with: ap skills add <owner>/<repo>")); continue; }
          for (const s of skills) console.log(`${cyan(s.name)} ${dim(`[${s.source}]`)} ${s.description.slice(0, 90)}`);
          if (config.light) console.log(dim("(--light profile: skills are not injected into the prompt)"));
          continue;
        }
        case "mcp": {
          const { mcpStatus, mcpServerSpecs, reloadMcp } = await import("./mcp.ts");
          if (rest[0] === "reload") {
            console.log(dim("reloading MCP servers…"));
            await reloadMcp(config, (m) => console.log(yellow(`⚠ ${m}`)));
            clearPromptSnapshots(session.id);
            const st = mcpStatus();
            const ok = st.filter((s) => s.ok).length;
            console.log(dim(`MCP reload done · ${ok}/${st.length} up — tool schemas changed (provider cache miss expected)`));
            continue;
          }
          await mcpReady;
          const st = mcpStatus();
          if (!st.length) {
            const configured = Object.keys(mcpServerSpecs(config)).length;
            console.log(dim(configured
              ? "MCP servers configured but not connected (light profile?)"
              : "no MCP servers — add with: ap mcp add <name> <command...> (or a project .mcp.json)"));
            continue;
          }
          for (const s of st) {
            const mark = s.ok ? green("●") : red("●");
            const meta = s.ok ? dim(` · ${s.tools.length} tools`) : ` ${red(s.error ?? "failed")}`;
            console.log(`${mark} ${cyan(s.name)} ${dim(`[${s.transport}]`)}${s.serverName ? dim(` ${s.serverName}`) : ""}${meta}`);
            for (const t of s.tools) {
              console.log(dim(`   ${t.canonical}${t.readOnly ? " (ro)" : ""} — ${t.description.replace(/\s+/g, " ").slice(0, 70)}`));
            }
          }
          console.log(dim("/mcp reload — reconnect and rebuild tool list"));
          continue;
        }
        case "undo": {
          const cps = cp.list(2);
          if (cps.length < 2) { console.log(dim("no earlier checkpoint to restore")); continue; }
          const target = cps[1]!;
          const r = cp.restore(target.hash);
          console.log(r ? dim(`restored checkpoint ${target.hash} (${target.label})`) : dim("restore failed"));
          continue;
        }
        case "restore": {
          if (!rest[0]) { console.log(dim("usage: /restore <hash> (see /checkpoints)")); continue; }
          const r = cp.restore(rest[0]);
          console.log(r ? dim(`restored checkpoint ${rest[0]}`) : dim("restore failed — check the hash"));
          continue;
        }
        case "checkpoints": {
          const cps = cp.list();
          if (!cps.length) { console.log(dim(cp.available() ? "no checkpoints yet (created after mutating turns)" : "checkpoints unavailable (light profile, disabled, or git missing)")); continue; }
          cps.forEach((c, i) => console.log(`${i === 0 ? cyan("▸") : " "} ${c.hash} ${dim(c.label)}`));
          continue;
        }
        case "diff": {
          const arg = rest[0];
          const { isWorkingDiffArg, isBranchDiffArg, workingDiff, branchDiff, diffForPrompt } = await import("./git.ts");
          if (isWorkingDiffArg(arg)) {
            const d = workingDiff(config);
            console.log(d ? diffForPrompt(d, 30_000) : dim("(clean working tree)"));
            continue;
          }
          if (isBranchDiffArg(arg)) {
            const r = branchDiff(config, arg!);
            if (!r.ok) { console.log(dim(r.out)); continue; }
            console.log(r.out ? diffForPrompt(r.out, 30_000) : dim(`(no diff vs ${arg})`));
            continue;
          }
          const n = Number(arg) || 1;
          console.log(cp.diff(n));
          continue;
        }
        case "worktree": {
          const sub = rest[0];
          const gitOk = Bun.spawnSync(["git", "-C", originalCwd, "rev-parse", "--git-dir"], { stdout: "ignore", stderr: "ignore" }).exitCode === 0;
          if (!gitOk) { console.log(dim("not a git repository — worktrees need one")); continue; }
          if (sub === "list") {
            const r = Bun.spawnSync(["git", "-C", originalCwd, "worktree", "list"], { stdout: "pipe", stderr: "pipe" });
            console.log(dim(r.stdout?.toString().trim() || "none"));
          } else if (sub === "back") {
            config.cwd = originalCwd;
            cp = new Checkpoints(config, cpSessionId); // work-tree is snapshotted at construction
            console.log(dim(`cwd → ${originalCwd}`));
          } else if (sub === "merge") {
            const slug = rest[1];
            if (!slug) { console.log(dim("usage: /worktree merge <slug>")); continue; }
            config.cwd = originalCwd;
            const m = Bun.spawnSync(["git", "-C", originalCwd, "merge", `ap/${slug}`], { stdout: "pipe", stderr: "pipe" });
            const out = (m.stdout?.toString() ?? "") + (m.stderr?.toString() ?? "");
            console.log(dim(out.trim().slice(0, 400) || "merged"));
            if (m.exitCode === 0) {
              const dir = join(config.dataDir, "worktrees", `${basename(originalCwd)}-${slug}`);
              Bun.spawnSync(["git", "-C", originalCwd, "worktree", "remove", "--force", dir], { stdout: "ignore", stderr: "ignore" });
              console.log(dim(`worktree removed · branch ap/${slug} kept`));
            }
          } else if (sub && sub !== "new") {
            console.log(dim("usage: /worktree <new <slug> | list | back | merge <slug>>"));
          } else {
            const slug = (sub === "new" ? rest[1] : rest[0])?.replace(/[^\w-]/g, "-").toLowerCase();
            if (!slug) { console.log(dim("usage: /worktree new <slug>")); continue; }
            const dir = join(config.dataDir, "worktrees", `${basename(originalCwd)}-${slug}`);
            const r = Bun.spawnSync(["git", "-C", originalCwd, "worktree", "add", dir, "-b", `ap/${slug}`], { stdout: "pipe", stderr: "pipe" });
            if (r.exitCode !== 0) { console.log(dim((r.stderr?.toString() ?? "worktree add failed").trim().slice(0, 300))); continue; }
            config.cwd = dir;
            cp = new Checkpoints(config, `${cpSessionId}-${slug}`); // own chain per worktree
            console.log(dim(`worktree ready: branch ap/${slug} · cwd → ${dir}`));
            console.log(dim(`/worktree back to return · /worktree merge ${slug} when done`));
          }
          continue;
        }
        case "compact": {
          if (session.history.length < 4) { console.log(dim("nothing worth compacting yet")); continue; }
          await compactNow("manual");
          continue;
        }
        case "archives": {
          const rows = listArchives(config.dataDir, 15);
          if (!rows.length) { console.log(dim("no compaction archives yet")); continue; }
          for (const a of rows) {
            console.log(dim(`${a.at.slice(0, 19)}  ${a.oldId} → ${a.newId}  ${a.summaryChars}c  ${a.reason}`));
          }
          console.log(dim("restore with /restore-context <oldId|newId>"));
          continue;
        }
        case "restore-context": {
          const id = rest[0];
          if (!id) { console.log(dim("usage: /restore-context <sessionId>  (see /archives)")); continue; }
          const note = restoreContextNote(config.dataDir, id);
          if (!note) { console.log(dim("session not found or empty")); continue; }
          session.append({ role: "user", content: note });
          session.append({ role: "assistant", content: "Restored context noted — ask me to use it." });
          console.log(dim(`restored context from ${id} (${note.length} chars) into current session`));
          continue;
        }
        case "rewind": {
          const n = Math.max(1, Math.min(20, parseInt(rest[0] ?? "1", 10) || 1));
          const dropped = session.rewind(n);
          if (!dropped) { console.log(dim("nothing to rewind")); continue; }
          console.log(dim(`rewound ${n} user turn(s) · dropped ${dropped} message(s) · workspace unchanged (/undo for files)`));
          continue;
        }
        case "agent": {
          const { getAgent, discoverAgents } = await import("./agents.ts");
          const sub = rest[0]?.toLowerCase();
          if (sub === "clear") {
            if (savedAgent) {
              config.toolFilter = savedAgent.toolFilter;
              provider = savedAgent.provider;
              savedAgent = null;
              agentRoleBody = null;
              clearPromptSnapshots(session.id);
              console.log(dim("agent profile cleared"));
            } else console.log(dim("no agent profile applied"));
            continue;
          }
          if (!sub) {
            const defs = discoverAgents(config);
            if (!defs.length) { console.log(dim("no agent profiles — add .ap/agents/<name>.md")); continue; }
            for (const a of defs) console.log(`${cyan(a.name)} ${dim(a.description.slice(0, 80))}`);
            if (agentRoleBody) console.log(dim("(a profile is active — /agent clear to drop it)"));
            continue;
          }
          const name = sub === "use" ? rest[1] : rest[0];
          if (!name) { console.log(dim("usage: /agent <name> | /agent use <name> | /agent clear")); continue; }
          const def = getAgent(config, name);
          if (!def) {
            const defs = discoverAgents(config);
            console.log(dim(`unknown agent "${name}"${defs.length ? ` — available: ${defs.map((a) => a.name).join(", ")}` : ""}`));
            continue;
          }
          if (!savedAgent) {
            savedAgent = { toolFilter: config.toolFilter, provider };
          }
          config.toolFilter = def.tools?.length ? def.tools : undefined;
          agentRoleBody = def.body?.trim() || null;
          if (def.model) {
            try {
              const slash = def.model.indexOf("/");
              const pfx = slash > 0 ? def.model.slice(0, slash) : "";
              if (pfx && config.providers[pfx]) {
                provider = resolveProvider(config, { ...flags, provider: pfx, model: def.model.slice(slash + 1) } as any);
              } else {
                provider = resolveProvider(config, { ...flags, model: def.model } as any);
              }
            } catch (e) {
              console.log(dim(`model override failed: ${(e as Error).message}`));
            }
          }
          clearPromptSnapshots(session.id);
          console.log(dim(`agent → ${def.name}${def.tools?.length ? ` · tools ${def.tools.join(",")}` : ""}${def.model ? ` · ${def.model}` : ""} (prefix cache miss ok)`));
          continue;
        }
        case "commit": {
          const { gitState, workingDiff, diffForPrompt, cleanCommitMessage, createBranch, commitAll, slugifyBranch } =
            await import("./git.ts");
          const flagsIn = new Set(rest.filter((a) => a.startsWith("--")));
          const stagedOnly = flagsIn.has("--staged");
          const sign = flagsIn.has("--sign");
          const msgParts = rest.filter((a) => !a.startsWith("--"));
          const st = gitState(config);
          if (!st.repo) { console.log(dim("not a git repository")); continue; }
          if (!stagedOnly && !st.dirty.length) { console.log(dim("nothing to commit — working tree clean")); continue; }

          // Protected branch: offer a feature branch instead of committing to it.
          if (st.protectedBranch) {
            console.log(dim(`on protected branch "${st.branch}" — AP won't commit here directly.`));
            process.stdout.write(dim(`create branch ${slugifyBranch(msgParts.join(" ") || history[history.length - 2] || "work")} and commit there? [y/N] `));
            const ans = await readKey(["y", "n"]);
            console.log(dim(ans === "y" ? "yes" : "no"));
            if (ans !== "y") { console.log(dim("aborted — switch branches yourself, then /commit")); continue; }
            const name = slugifyBranch(msgParts.join(" ") || history[history.length - 2] || "work");
            const b = createBranch(config, name);
            if (!b.ok) { console.log(dim(`branch failed: ${b.out.slice(0, 200)}`)); continue; }
            console.log(dim(`branch → ${name}`));
          }

          // Message: explicit argument wins; otherwise the model drafts one.
          let msg = msgParts.join(" ").trim();
          if (!msg) {
            spinner.start("drafting commit message…");
            try {
              const diff = diffForPrompt(workingDiff(config));
              let draft = "";
              await streamChat(
                provider,
                [
                  { role: "system", content: "Write a git commit message for this diff. One imperative subject line under 72 chars (no period, no type prefix unless the repo clearly uses one), then an optional short body explaining WHY. Output only the message." },
                  { role: "user", content: diff || `Files changed:\n${st.dirty.join("\n")}` },
                ],
                [], (d) => { draft += d; }, undefined, undefined, undefined, config.streamIdleSeconds * 1000,
              );
              msg = cleanCommitMessage(draft);
            } catch (e) {
              console.log(dim(`draft failed: ${(e as Error).message}`));
            } finally { spinner.stop(); }
          }
          if (!msg) { console.log(dim("no message — /commit <your message> to write one yourself")); continue; }

          console.log(dim(`${st.dirty.length} file${st.dirty.length === 1 ? "" : "s"} · branch ${gitState(config).branch}`));
          for (const l of msg.split("\n")) console.log(l ? `  ${l}` : "");
          process.stdout.write(dim("commit this? [y/N] "));
          const ok = await readKey(["y", "n"]);
          console.log(dim(ok === "y" ? "yes" : "no"));
          if (ok !== "y") { console.log(dim("aborted — nothing committed")); continue; }
          const r = commitAll(config, msg, { stagedOnly, sign });
          console.log(dim(r.ok ? `committed ${r.out}${sign ? " (signed)" : ""}${stagedOnly ? " (staged only)" : ""} (not pushed)` : `commit failed: ${r.out.slice(0, 300)}`));
          continue;
        }
        case "pr": {
          if (config.light) { console.log(dim("/pr is a full-profile feature")); continue; }
          const { gitState, prPromptMaterial, createPullRequest, cleanCommitMessage } = await import("./git.ts");
          const flagsIn = new Set(rest.filter((a) => a.startsWith("--") && !a.startsWith("--base")));
          const draft = flagsIn.has("--draft");
          let base: string | undefined;
          const titleParts: string[] = [];
          for (let i = 0; i < rest.length; i++) {
            const a = rest[i]!;
            if (a === "--base") { base = rest[++i]; continue; }
            if (a === "--draft") continue;
            if (a.startsWith("--")) continue;
            titleParts.push(a);
          }
          const st = gitState(config);
          if (!st.repo) { console.log(dim("not a git repository")); continue; }
          if (st.protectedBranch) {
            console.log(dim(`on protected branch "${st.branch}" — switch to an ap/… or feature branch first`));
            continue;
          }
          const mat = prPromptMaterial(config, base);
          let title = titleParts.join(" ").trim();
          let body = "";
          if (!title) {
            spinner.start("drafting PR…");
            try {
              let draftText = "";
              await streamChat(
                provider,
                [
                  {
                    role: "system",
                    content:
                      "Draft a GitHub pull request. First line: title under 72 chars (imperative). Then a blank line. Then a short markdown body with ## Summary (2-3 bullets) and ## Test plan (checklist). Output only the PR text.",
                  },
                  {
                    role: "user",
                    content: `Branch ${mat.head} → ${mat.base}\n\nCommits:\n${mat.commits || "(none)"}\n\nDiff:\n${mat.diff || "(empty)"}`,
                  },
                ],
                [], (d) => { draftText += d; }, undefined, undefined, undefined, config.streamIdleSeconds * 1000,
              );
              const lines = draftText.trim().split(/\r?\n/);
              title = cleanCommitMessage(lines[0] ?? "").split("\n")[0] ?? "";
              body = lines.slice(1).join("\n").replace(/^\s*\n/, "").trim();
            } catch (e) {
              console.log(dim(`draft failed: ${(e as Error).message}`));
            } finally { spinner.stop(); }
          }
          if (!title) { console.log(dim("no title — /pr <title> or let the model draft when a provider is configured")); continue; }
          if (!body) body = `## Summary\n- ${title}\n\n## Test plan\n- [ ] verified locally`;
          console.log(dim(`${mat.head} → ${mat.base}${draft ? " (draft)" : ""}`));
          console.log(`  ${title}`);
          for (const l of body.split("\n").slice(0, 12)) console.log(l ? `  ${l}` : "");
          process.stdout.write(dim("create this PR? [y/N] "));
          const ok = await readKey(["y", "n"]);
          console.log(dim(ok === "y" ? "yes" : "no"));
          if (ok !== "y") { console.log(dim("aborted — nothing created")); continue; }
          const r = createPullRequest(config, { title, body, base: mat.base, draft });
          console.log(dim(r.ok ? r.out : `pr failed: ${r.out.slice(0, 400)}`));
          continue;
        }
        case "spawn": {
          if (config.light) { console.log(dim("/spawn is a full-profile feature")); continue; }
          const task = rest.join(" ").trim();
          if (!task) { console.log(dim("usage: /spawn <task> — detaches ap run in tmux (unix)")); continue; }
          const { tmuxSpawn, tmuxAvailable, tmuxMissingHint } = await import("./tmux.ts");
          if (!tmuxAvailable()) { console.log(dim(tmuxMissingHint())); continue; }
          const r = tmuxSpawn(config, task);
          console.log(dim(r.ok ? r.out : `spawn failed: ${r.out}`));
          continue;
        }
        case "tmux": {
          if (config.light) { console.log(dim("/tmux is a full-profile feature")); continue; }
          const { tmuxList, tmuxLayout, tmuxCapture, tmuxAvailable, tmuxMissingHint } = await import("./tmux.ts");
          if (!tmuxAvailable()) { console.log(dim(tmuxMissingHint())); continue; }
          const sub = rest[0] ?? "list";
          if (sub === "layout") {
            const r = tmuxLayout(config);
            console.log(dim(r.ok ? r.out : r.out));
          } else if (sub === "capture") {
            if (!rest[1]) { console.log(dim("usage: /tmux capture <session> [lines]")); continue; }
            const r = tmuxCapture(rest[1], Number(rest[2]) || 80);
            console.log(r.ok ? r.out : dim(r.out));
          } else {
            const r = tmuxList();
            console.log(r.ok ? r.out : dim(r.out));
          }
          continue;
        }
        case "share": {
          if (config.light) { console.log(dim("/share is a full-profile feature")); continue; }
          if (!session.history.length) { console.log(dim("nothing to share yet")); continue; }
          try {
            const { exportSessionHtml } = await import("./shareview.ts");
            const { openInBrowser } = await import("./planview.ts");
            const p = exportSessionHtml(config.dataDir, session.id, session.history, `${provider.name}/${provider.model}`, config.cwd);
            openInBrowser(p);
            console.log(dim(`transcript → ${p} (opened in browser) — one self-contained file, host or send it anywhere`));
          } catch (e) { console.log(dim(`share failed: ${(e as Error).message}`)); }
          continue;
        }
        default:
          console.log(dim(`unknown command ${input.split(/\s+/)[0]} — type / to see commands`));
          continue;
      }
      }
    }

    let userText = input;

    // Background-task results queued since the last turn are folded into this
    // user message (the loop.ts pending pattern): the model learns outcomes
    // the next time it runs, never mid-turn, and history stays append-only.
    // @file mentions: inline referenced files so the model skips a read turn.
    if (!config.light && /@[\w./\\-]+/.test(userText)) {
      const attached: string[] = [];
      for (const m of userText.match(/@([\w./\\-]+)/g) ?? []) {
        const rel = m.slice(1);
        const p = resolve(config.cwd, rel);
        try {
          if (existsSync(p)) {
            const body = readFileSync(p, "utf8");
            if (!body.includes("\0")) {
              userText += `\n\n<file path="${rel}">\n${body.slice(0, 8000)}${body.length > 8000 ? "\n[truncated]" : ""}\n</file>`;
              attached.push(rel);
            }
          }
        } catch {}
      }
      if (attached.length) process.stdout.write(dim(`[attached: ${attached.join(", ")}]\n`));
    }

    // Background-task results are folded in AFTER @file expansion, never
    // before: subagent text is model output, and an "@../../secret" inside it
    // would otherwise be expanded into context as if the USER had typed it.
    // Only what the user typed drives @-expansion.
    if (!config.light) {
      const { drainTaskNotes } = await import("./tasks.ts");
      const taskNotes = drainTaskNotes();
      if (taskNotes.length) {
        userText = `${taskNotes.map((n) => `<task-result>\n${n}\n</task-result>`).join("\n")}\n\n${userText}`;
        console.log(dim(`  folded ${taskNotes.length} background task result(s) into this turn`));
      }
      const steerNotes = drainSteerNotes();
      if (steerNotes.length) {
        userText = `${steerNotes.map((n) => `<steer>\n${n}\n</steer>`).join("\n")}\n\n${userText}`;
        console.log(dim(`  folded ${steerNotes.length} steer note(s) into this turn`));
      }
      const { drainChannelNotes } = await import("./channels.ts");
      const chNotes = drainChannelNotes();
      if (chNotes.length) {
        userText = `${chNotes.join("\n")}\n\n${userText}`;
        console.log(dim(`  folded ${chNotes.length} channel note(s) into this turn`));
      }
    }

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

    activeTools = new Map();
    const totals = { prompt: 0, cached: 0, completion: 0, steps: 0 };
    const t0 = performance.now();
    turnT0 = t0;
    turnOut = 0;
    steering = false;
    steerDraft = "";

    const spinnerLabel = () => {
      if (activeTools.size === 0) return "thinking";
      if (activeTools.size === 1) {
        const t = [...activeTools.values()][0]!;
        const ts = ((performance.now() - t.t0) / 1000).toFixed(1);
        return `${t.label} ${ts}s`;
      }
      return `${activeTools.size} tools running`;
    };

    const writeBoth = (s: string) => {
      fullBuf += s; compactBuf += s;
      process.stdout.write(s);
      printedBuf += s;
    };

    // One keypress: y (allow once) / a (always this session) / n·Enter·Esc (deny).
    const readPermitKey = (): Promise<"y" | "n" | "a"> =>
      new Promise((res) => {
        const done = (v: "y" | "n" | "a") => {
          process.stdin.removeListener("keypress", onKey);
          res(v);
        };
        const onKey = (_s: string, key: any) => {
          if (!key) return;
          if (key.ctrl && key.name === "c") { done("n"); return; } // global handler aborts the turn
          if (key.ctrl) return;
          if (key.name === "y") done("y");
          else if (key.name === "a") done("a");
          else if (key.name === "n" || key.name === "return" || key.name === "enter" || key.name === "escape") done("n");
        };
        process.stdin.on("keypress", onKey);
      });

    const permit: PermitFn = (req) => {
      const key = req.path
        ? dirname(resolve(req.path)).toLowerCase()
        : (req.detail.split(/\s+/)[0] ?? req.detail).toLowerCase();
      const result = permitChain.then(async () => {
        if (ctrl?.signal.aborted) return false;
        if (sessionAllows.has(key)) return true;
        spinner.stop();
        permitBusy = true;
        try {
          const sbHint = hinted.has("sandbox") ? "" : dim(" · /sandbox shows the rules");
          hinted.add("sandbox");
          writeBoth(`  ${yellow("?")} ${req.action}: ${req.detail} ${dim("— allow? [y/N/a=always]")}${sbHint} `);
          const ans = await readPermitKey();
          writeBoth(dim(ans === "a" ? "always\n" : ans === "y" ? "yes\n" : "no\n"));
          if (ans === "a") { sessionAllows.add(key); return true; }
          return ans === "y";
        } finally {
          permitBusy = false;
        }
      });
      permitChain = result.catch(() => { permitBusy = false; });
      return result;
    };

    let turnMutated = false;
    const emit = (e: AgentEvent) => {
      spinner.stop();
      if (e.type === "text" || e.type === "reasoning") turnOut += e.delta.length;
      let label: string | undefined;
      if (e.type === "tool_start") {
        label = toolLabel(e.name, e.args);
        activeTools.set(e.id, { label, t0: performance.now() });
      } else if (e.type === "tool_end") {
        label = activeTools.get(e.id)?.label ?? e.name;
        activeTools.delete(e.id);
        if (!e.error && getTool(e.name)?.readOnly === false) turnMutated = true;
      } else if (e.type === "turn_end") {
        totals.steps++;
        if (e.usage) {
          lastUsage = e.usage;
          totals.prompt += e.usage.prompt;
          totals.cached += e.usage.cached ?? 0;
          totals.completion += e.usage.completion;
          spend.prompt += e.usage.prompt;
          spend.cached += e.usage.cached ?? 0;
          spend.completion += e.usage.completion;
        }
      }

      const hideReason = config.showReasoning === "off";
      const fullChunk = (e.type === "reasoning" && hideReason) ? "" : fullR.feed(e, label);
      const compactChunk = (e.type === "reasoning" && hideReason) ? "" : compactR.feed(e, label);
      fullBuf += fullChunk;
      compactBuf += compactChunk;
      const live = verbose ? fullChunk : compactChunk;
      if (live) {
        // While the viewer owns the screen, HOLD output instead of writing it:
        // a write would land on the alternate screen buffer (corrupting the
        // viewer and making it flicker) and then be lost on exit. Held text is
        // flushed verbatim when the viewer closes, so the transcript is whole.
        if (watching) held += live;
        else process.stdout.write(live);
        printedBuf += live;
      }

      if (e.type === "tool_end" && !e.error && e.name === "agent" && !hinted.has("agents")) {
        hinted.add("agents");
        writeBoth(`  ${dim("  /agents lists this session's subagents")}\n`);
      }

      if (e.type === "tool_start" || e.type === "tool_end") spinner.start(spinnerLabel());
      else if (e.type === "text" && planMode && fullR.planTruncated) spinner.start("writing plan…");

      // Keep the pinned bar and the cross-process snapshot current as work
      // happens — subagent starts/ends are exactly what the user watches for.
      if (e.type === "tool_start" || e.type === "tool_end" || e.type === "subline") publishStatus(true);
    };

    spinner.start("thinking");
    // Pin the status to the bottom for the whole turn: output scrolls above
    // it, so the model/ctx/cost/agents line never scrolls out of sight.
    publishStatus(true, true);
    // Heartbeat: long model thinking emits no tool events, so without this the
    // snapshot goes stale and `ap watch` shows "17s ago" on a working process.
    const beat = setInterval(() => publishStatus(true, true), 1000);
    if (process.stdin.isTTY) process.stdin.setRawMode(true);
    let finalText = "";
    try {
      await mcpReady; // no-op after the first turn
      finalText = await runTurn(config, provider, session, userText, emit, ctrl.signal, {
        permit,
        extra: effort ? { reasoning_effort: effort } : undefined,
        systemOverride: agentRoleBody
          ? `${buildSystemPrompt(config)}\n\nRole — you are acting as a named agent:\n${agentRoleBody}`
          : undefined,
      });
    } catch {
      // error already emitted
    } finally {
      spinner.stop();
      clearInterval(beat);
      publishStatus(false, true);
      ctrl = null;
      turnT0 = 0;
      // Resolve pricing for the current model in the background (disk-cached
      // catalog; re-resolves only after a model switch).
      const modelKey = `${provider.name}/${provider.model}`;
      if (!config.light && pricedFor !== modelKey) {
        pricedFor = modelKey;
        pricing = null;
        import("./models.ts")
          .then(async ({ loadCatalog, modelPricing }) => {
            pricing = modelPricing(await loadCatalog(config.dataDir), provider.name, provider.model);
          })
          .catch(() => {});
      }
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      const cached = totals.cached ? ` (${totals.cached} cached)` : "";
      const hit = totals.cached && totals.prompt
        ? ` · cache ${Math.min(100, Math.round((totals.cached / totals.prompt) * 100))}%`
        : "";
      // Context drift belongs on the line people already read after each turn.
      let ctxNote = "";
      let costNote = "";
      try {
        let chars = buildSystemPrompt(config).length;
        for (const m of session.history) chars += JSON.stringify(m).length;
        const pct = Math.round((chars / config.contextBudgetChars) * 100);
        if (pct >= 1) ctxNote = ` · ctx ${pct}%`;
        if (pct >= 60 && !hinted.has("compact")) {
          hinted.add("compact");
          ctxNote += " — /compact frees context";
        }
      } catch {}
      if (pricing && (totals.prompt || totals.completion)) {
        const cachedTok = totals.cached;
        const usd =
          ((totals.prompt - cachedTok) * pricing.input +
            cachedTok * (pricing.cacheRead ?? pricing.input) +
            totals.completion * pricing.output) / 1e6;
        if (usd > 0) {
          costNote = usd < 0.01 ? ` · ~$${usd.toFixed(4)}` : usd < 1 ? ` · ~$${usd.toFixed(3)}` : ` · ~$${usd.toFixed(2)}`;
        }
      }
      writeBoth(dim(`\n${totals.steps} step${totals.steps === 1 ? "" : "s"} · ↑${totals.prompt}${cached}${hit} ↓${totals.completion} · ${secs}s${costNote}${ctxNote}\n`));
      if (pendingSteerCount() && !hinted.has("steer-pending")) {
        hinted.add("steer-pending");
        writeBoth(dim(`  ${pendingSteerCount()} steer note(s) waiting — will fold into your next message\n`));
      }
    }

    if (turnMutated && cp.available()) {
      const hash = cp.commit(input);
      if (hash) writeBoth(dim(`✓ checkpoint ${hash} · /undo to revert\n`));
    }

    if (config.mode === "plan" && finalText.trim()) {
      lastPlan = finalText;
      // Export only when the plan outgrew the terminal, or the user asked
      // for a page — short plans just print inline. Light profile: never.
      const wantsHtml = /\b(html|browser|page)\b/i.test(input);
      if (!config.light && (fullR.planTruncated || wantsHtml)) {
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

    // Auto-compaction (opencode parity): near the trim budget, summarize into
    // a fresh session instead of silently eliding old tool results forever.
    if (!config.light && config.autoCompact !== "off" && session.history.length >= 4) {
      try {
        let chars = buildSystemPrompt(config).length;
        for (const m of session.history) chars += JSON.stringify(m).length;
        if (chars >= config.contextBudgetChars * 0.85) {
          console.log(dim(`context ${Math.round((chars / config.contextBudgetChars) * 100)}% of budget — auto-compacting ("autoCompact": "off" disables)`));
          await compactNow("auto");
        }
      } catch {}
    }
  }
}
