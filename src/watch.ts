// Interactive process viewer (/watch, ctrl+w, ap watch). Opens in the SAME
// terminal on the alternate screen buffer, so the transcript underneath is
// restored untouched on exit — like less/htop, not a cleared screen.
//
// Two levels: a list of processes with their subagents, and a detail pane
// showing one subagent's task and full output. Deliberately READ-ONLY —
// there is no prompt here; Esc hands you back to the session that owns input.
//
// Rendering discipline (the flicker fix): NEVER clear the whole screen per
// frame. `\x1b[2J` on a 700ms timer makes the screen strobe, and it strobes
// worse when anything else writes. Instead each row is repainted in place
// with erase-to-end-of-line, and an unchanged frame is not written at all.
//
// Keys — list: ←/→ process · ↑/↓ subagent · Enter detail · r refresh · Esc exit
//        detail: ↑/↓ scroll · Esc back
import { currentTheme, fitLine, visibleLen } from "./theme.ts";
import { formatSnapshot, readLive, type LiveAgent, type LiveSnapshot } from "./live.ts";

const ALT_ON = "\x1b[?1049h";
const ALT_OFF = "\x1b[?1049l";
const HIDE = "\x1b[?25l";
const SHOW = "\x1b[?25h";
const R = "\x1b[0m";
const INV = "\x1b[7m";

export interface WatchIO {
  write: (s: string) => unknown;
  rows?: number;
  columns?: number;
  isTTY?: boolean;
}

export interface ViewState {
  /** Index into the snapshot list. */
  proc: number;
  /** Index into the selected process's agents (-1 = none highlighted). */
  agent: number;
  /** Detail pane open for the selected agent. */
  detail: boolean;
  /** First visible line of the detail pane. */
  scroll: number;
}

/**
 * Tail of a background process's log, read straight off disk. The viewer may
 * be a DIFFERENT process from the one that started the job, so it cannot ask
 * the owner — but the log path is in the snapshot and the file is shared.
 * Bounded read: these logs can grow without limit.
 */
export function readLogTail(path: string, maxLines = 200, maxBytes = 64_000): string {
  try {
    const { readFileSync, statSync } = require("node:fs") as typeof import("node:fs");
    const size = statSync(path).size;
    const fd = require("node:fs").openSync(path, "r");
    try {
      const start = Math.max(0, size - maxBytes);
      const buf = Buffer.alloc(Math.min(size, maxBytes));
      require("node:fs").readSync(fd, buf, 0, buf.length, start);
      const text = buf.toString("utf8");
      const lines = text.split(/\r?\n/);
      if (start > 0) lines.shift(); // drop the partial first line
      return lines.slice(-maxLines).join("\n").trim() || "(log is empty)";
    } finally { require("node:fs").closeSync(fd); }
  } catch (e) {
    return `(log unavailable: ${(e as Error).message})`;
  }
}

const wrapText = (s: string, width: number): string[] => {
  const out: string[] = [];
  for (const raw of s.split(/\r?\n/)) {
    if (!raw.length) { out.push(""); continue; }
    for (let i = 0; i < raw.length; i += width) out.push(raw.slice(i, i + width));
  }
  return out;
};

/**
 * Render one frame as an array of exactly `rows` lines. Pure, so the whole
 * layout — selector row, agent list, detail pane, bottom-right legend — is
 * assertable without a terminal.
 */
export function renderFrame(
  snaps: LiveSnapshot[],
  st: ViewState,
  rows: number,
  cols: number,
  now = Date.now(),
  self?: number,
): string[] {
  const dim = currentTheme().dim;
  const accent = currentTheme().accent;
  const width = Math.max(20, cols - 1);
  const out: string[] = [];
  const title = ` ap watch · ${snaps.length} process${snaps.length === 1 ? "" : "es"} `;
  out.push(`${accent}${title}${R}${dim}${"─".repeat(Math.max(0, width - visibleLen(title)))}${R}`);

  let legend = `${dim}←/→ process · ↑/↓ subagent · Enter detail · Esc back${R}`;

  if (!snaps.length) {
    out.push("");
    out.push(`${dim}  No ap process is publishing status.${R}`);
    out.push(`${dim}  Start one in another terminal, or run a turn here — subagents,${R}`);
    out.push(`${dim}  background tasks and flows appear as they start.${R}`);
    legend = `${dim}r refresh · Esc back${R}`;
  } else {
    const s = snaps[Math.min(Math.max(st.proc, 0), snaps.length - 1)]!;
    const agents = s.agents ?? [];
    const sel = agents.length ? Math.min(Math.max(st.agent, 0), agents.length - 1) : -1;

    const procs = s.procs ?? [];
    // Agents and detached shell processes share one selection list: both are
    // "things running in the background", and a user who started either wants
    // the same drill-down.
    const items: { kind: "agent" | "proc"; i: number }[] = [
      ...agents.map((_, i) => ({ kind: "agent" as const, i })),
      ...procs.map((_, i) => ({ kind: "proc" as const, i })),
    ];
    const pick = items.length ? items[Math.min(Math.max(st.agent, 0), items.length - 1)]! : null;

    if (st.detail && pick) {
      // Detail keeps the SAME chrome as the list — process tabs and the
      // session line — so entering an item feels like drilling in, not
      // jumping to an unrelated screen.
      const tabs = snaps.map((sn, i) => {
        const label = ` ${sn.pid === self ? "this session" : `pid ${sn.pid}`} `;
        return i === st.proc ? `${INV}${label}${R}` : `${dim}${label}${R}`;
      });
      out.push(tabs.join(`${dim}│${R}`));
      const body: string[] = [];
      if (pick.kind === "proc") {
        const p = procs[pick.i]!;
        out.push(`${accent} ▸ process ${p.pid} ${R}${dim} ${p.alive ? "running" : "exited"}${R}`);
        out.push("");
        body.push(`${dim}command${R}`);
        body.push(...wrapText(p.cmd, width - 2).map((l) => `  ${l}`));
        body.push("");
        body.push(`${dim}log · ${p.log}${R}`);
        body.push(...wrapText(readLogTail(p.log), width - 2).map((l) => `  ${l}`));
      } else {
        const a = agents[pick.i]!;
        const secs = Math.round((now - a.startedAt) / 1000);
        out.push(`${accent} ◇ subagent #${a.id} ${R}${dim} ${a.status} · ${a.steps} steps · ${secs}s${a.background ? " · background" : ""}${R}`);
        out.push("");
        body.push(`${dim}task${R}`);
        body.push(...wrapText(a.fullTask || a.label, width - 2).map((l) => `  ${l}`));
        body.push("");
        // A finished agent shows its answer; a RUNNING one shows the live
        // reasoning/text streaming out of the child, so it is never a black box.
        if (a.result) {
          body.push(`${dim}output${R}`);
          body.push(...wrapText(a.result, width - 2).map((l) => `  ${l}`));
        } else if (a.live) {
          body.push(`${dim}thinking (live)${R}`);
          body.push(...wrapText(a.live, width - 2).map((l) => `  ${l}`));
        } else {
          body.push(`${dim}output${R}`);
          body.push(`${dim}  (started — nothing streamed yet)${R}`);
        }
      }
      // Viewport sized to everything left on screen, so a long output uses
      // the whole window instead of a few cramped rows.
      const room = Math.max(1, rows - out.length - 1);
      const maxStart = Math.max(0, body.length - room);
      const start = Math.min(Math.max(st.scroll, 0), maxStart);
      out.push(...body.slice(start, start + room));
      legend = body.length > room
        ? `${dim}↑/↓ scroll ${start + 1}-${Math.min(start + room, body.length)}/${body.length} · ←/→ process · Esc back${R}`
        : `${dim}←/→ process · Esc back to the list${R}`;
    } else {
      const tabs = snaps.map((sn, i) => {
        const label = ` ${sn.pid === self ? "this session" : `pid ${sn.pid}`} `;
        return i === st.proc ? `${INV}${label}${R}` : `${dim}${label}${R}`;
      });
      out.push(tabs.join(`${dim}│${R}`));
      out.push("");
      const [head, cwdLine, ...restLines] = formatSnapshot({ ...s, agents: [] }, now);
      out.push(head!);
      out.push(cwdLine!);
      for (const l of restLines) out.push(l); // flow line, if any
      out.push("");
      if (!items.length) {
        out.push(`${dim}  nothing running here yet — subagents (agent tool),${R}`);
        out.push(`${dim}  background shell processes (bash background:true) and${R}`);
        out.push(`${dim}  workflows all appear in this list as they start.${R}`);
        legend = `${dim}←/→ process · r refresh · Esc back${R}`;
      } else {
        let row = 0;
        for (const a of agents) {
          const secs = Math.round((now - a.startedAt) / 1000);
          const line = ` ◇ #${a.id} [${a.status}]${a.background ? " &" : ""} ${a.label.slice(0, Math.max(10, width - 34))} · ${a.steps} steps · ${secs}s `;
          out.push(row === st.agent ? `${INV}${fitLine(line, width - 1)}${R}` : `${dim}${fitLine(line, width - 1)}${R}`);
          row++;
        }
        for (const p of procs) {
          const size = p.bytes == null ? "no log" : p.bytes < 1024 ? `${p.bytes}B` : `${Math.round(p.bytes / 1024)}KB`;
          const line = ` ▸ pid ${p.pid} [${p.alive ? "running" : "exited"}] ${p.cmd.replace(/\s+/g, " ").slice(0, Math.max(10, width - 34))} · ${size} `;
          out.push(row === st.agent ? `${INV}${fitLine(line, width - 1)}${R}` : `${dim}${fitLine(line, width - 1)}${R}`);
          row++;
        }
        // Preview the selection — enough to tell runs apart without opening
        // the detail pane.
        const first = pick?.kind === "agent"
          ? (agents[pick.i]!.result ?? "").split(/\r?\n/).find((l) => l.trim())
          : pick
            ? readLogTail(procs[pick.i]!.log, 1).split(/\r?\n/).find((l) => l.trim())
            : undefined;
        if (first) {
          out.push("");
          out.push(`${dim}  ${fitLine(first, width - 4)}${R}`);
          out.push(`${dim}  Enter for the full output${R}`);
        }
      }
    }
  }

  // Pad, then place the legend flush RIGHT on the last row.
  while (out.length < rows - 1) out.push("");
  out.length = rows - 1;
  out.push(" ".repeat(Math.max(0, width - visibleLen(legend))) + legend);
  return out.map((l) => fitLine(l, width));
}

export interface WatchOpts {
  dataDir: string;
  io?: WatchIO;
  refreshMs?: number;
}

/**
 * Run the viewer until the user exits. Resolves once the alternate screen has
 * been torn down. Caller must have run readline.emitKeypressEvents(stdin).
 */
export function runWatch(opts: WatchOpts): Promise<void> {
  const io = opts.io ?? (process.stdout as any as WatchIO);
  const stdin = process.stdin;
  return new Promise((resolve) => {
    const st: ViewState = { proc: 0, agent: 0, detail: false, scroll: 0 };
    let timer: ReturnType<typeof setInterval> | null = null;
    let prev: string[] = [];
    const wasRaw = !!(stdin as any).isRaw;

    const draw = (force = false) => {
      const snaps = readLive(opts.dataDir);
      if (st.proc >= snaps.length) st.proc = Math.max(0, snaps.length - 1);
      const rows = io.rows ?? 24;
      const cols = io.columns ?? 80;
      const frame = renderFrame(snaps, st, rows, cols, Date.now(), process.pid);
      // Repaint only CHANGED rows, in place. No full clear → no flicker.
      let s = "";
      for (let i = 0; i < frame.length; i++) {
        if (!force && prev[i] === frame[i]) continue;
        s += `\x1b[${i + 1};1H\x1b[K${frame[i]}`;
      }
      if (force && frame.length < prev.length) {
        for (let i = frame.length; i < prev.length; i++) s += `\x1b[${i + 1};1H\x1b[K`;
      }
      prev = frame;
      if (s) { try { io.write(s); } catch {} }
    };

    const finish = () => {
      if (timer) clearInterval(timer);
      timer = null;
      stdin.removeListener("keypress", onKey);
      try { io.write(SHOW + ALT_OFF); } catch {}
      // Restore the caller's raw-mode state; the REPL keeps stdin raw between
      // turns and would otherwise lose its key handling.
      if (stdin.isTTY) { try { (stdin as any).setRawMode(wasRaw); } catch {} }
      resolve();
    };

    /** Selectable rows = subagents followed by background processes. */
    const itemCount = (snaps: LiveSnapshot[]) => {
      const s = snaps[Math.min(st.proc, Math.max(0, snaps.length - 1))];
      return (s?.agents?.length ?? 0) + (s?.procs?.length ?? 0);
    };

    const onKey = (_s: string, key: any) => {
      if (!key) return;
      if (key.ctrl && key.name === "c") return finish();
      if (key.name === "escape" || key.name === "q") {
        if (st.detail) { st.detail = false; st.scroll = 0; return draw(true); }
        return finish();
      }
      const snaps = readLive(opts.dataDir);
      const n = itemCount(snaps);
      if (key.name === "return" || key.name === "enter") {
        if (!st.detail && n) { st.detail = true; st.scroll = 0; draw(true); }
        return;
      }
      if (st.detail) {
        if (key.name === "up") { st.scroll = Math.max(0, st.scroll - 1); draw(); }
        else if (key.name === "down") { st.scroll++; draw(); }
        else if (key.name === "pageup") { st.scroll = Math.max(0, st.scroll - 10); draw(); }
        else if (key.name === "pagedown") { st.scroll += 10; draw(); }
        else if (key.name === "r") draw(true);
        return;
      }
      if (key.name === "right" || key.name === "tab") {
        if (snaps.length) { st.proc = (st.proc + 1) % snaps.length; st.agent = 0; draw(true); }
      } else if (key.name === "left") {
        if (snaps.length) { st.proc = (st.proc - 1 + snaps.length) % snaps.length; st.agent = 0; draw(true); }
      } else if (key.name === "down") {
        if (n) { st.agent = (st.agent + 1) % n; draw(); }
      } else if (key.name === "up") {
        if (n) { st.agent = (st.agent - 1 + n) % n; draw(); }
      } else if (key.name === "r") {
        draw(true);
      }
    };

    if (stdin.isTTY) { try { (stdin as any).setRawMode(true); } catch {} }
    stdin.resume();
    // Enter the alternate screen and clear it ONCE; every later frame is a
    // differential repaint.
    try { io.write(ALT_ON + HIDE + "\x1b[2J"); } catch {}
    draw(true);
    timer = setInterval(() => draw(), opts.refreshMs ?? 700);
    stdin.on("keypress", onKey);
  });
}
