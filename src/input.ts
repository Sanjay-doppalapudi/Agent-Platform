// Minimal raw-mode line reader with dropdown menus. Zero dependencies.
// "/" opens the command menu; "@" opens the file picker (lazy rg --files).
// Typing filters; ↑/↓ scroll (windowed); Enter/Tab select; Esc closes.
// ↑/↓ recall history when no menu is open.
// Caller must have run readline.emitKeypressEvents(process.stdin) once.

import { currentTheme, cursorGeometry, reflowRewind, visibleLen } from "./theme.ts";

const R = "\x1b[0m";
const DIM = () => currentTheme().dim; // theme-aware (mono/NO_COLOR → no codes)
const INV = "\x1b[7m";

export interface SlashCommand {
  name: string; // "/plan"
  desc: string;
  hasArg?: boolean;
}

interface MenuItem {
  label: string;
  desc: string;
  /** Text that replaces the active token when selected. */
  insert: string;
  /** Submit the buffer immediately on Enter (no-arg commands). */
  submit: boolean;
}

const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const MENU_ROWS = 8;

/** Hidden input for secrets — echoes `*` per char. */
export function readSecret(promptStr: string): Promise<string> {
  return new Promise((resolve) => {
    const stdin = process.stdin;
    if (stdin.isTTY) stdin.setRawMode(true);
    stdin.resume();
    let buf = "";
    process.stdout.write(promptStr);
    const onKey = (str: string, key: any) => {
      if (key?.name === "return" || key?.name === "enter") {
        stdin.removeListener("keypress", onKey);
        process.stdout.write("\n");
        if (stdin.isTTY) stdin.setRawMode(false);
        resolve(buf);
      } else if (key?.name === "backspace") {
        if (buf) { buf = buf.slice(0, -1); process.stdout.write("\b \b"); }
      } else if (key?.ctrl && key.name === "c") {
        process.stdout.write("\n");
        process.exit(1);
      } else if (typeof str === "string" && str.length > 0 && str >= " ") {
        buf += str;
        process.stdout.write("*");
      }
    };
    stdin.on("keypress", onKey);
  });
}

export function readLine(opts: {
  prompt: string; // may contain ANSI colors
  commands: SlashCommand[];
  history: string[];
  /** Lazy workspace file list for @-completion (called once, cached). */
  files?: () => string[];
  onCtrlO?: () => void;
  /** Status row rendered below the input line (may contain ANSI colors).
   *  A function is re-evaluated on every render (memoize if costly).
   *  Doubles as the frame's bottom edge, so it is drawn under menus too. */
  status?: string | (() => string);
  /** Frame top edge, redrawn above the input on every render (so it survives
   *  resizes and ctrl+o replays). Return "" to draw no frame. */
  frameTop?: () => string;
}): Promise<string | null> {
  const { prompt, commands, history } = opts;
  const promptLen = stripAnsi(prompt).length;

  return new Promise((resolve) => {
    const stdin = process.stdin;
    if (stdin.isTTY) stdin.setRawMode(true);
    stdin.resume();

    let buf = "";
    let menuIdx = 0;
    let menuClosed = false; // Esc pressed; reopens on next edit
    let histIdx = history.length;
    let fileCache: string[] | null = null;

    /** The token the menu operates on: {start, items} or null. */
    const activeMenu = (): { start: number; items: MenuItem[] } | null => {
      if (menuClosed) return null;
      // File picker: trailing @token (works mid-message).
      const fm = opts.files ? buf.match(/@([\w./\\-]*)$/) : null;
      if (fm) {
        if (!fileCache) {
          try { fileCache = opts.files!(); } catch { fileCache = []; }
        }
        const q = fm[1]!.toLowerCase();
        const items = fileCache
          .filter((f) => !q || f.toLowerCase().includes(q))
          .slice(0, 200)
          .map((f) => ({ label: f, desc: "", insert: `@${f} `, submit: false }));
        return items.length ? { start: buf.length - fm[0].length, items } : null;
      }
      // Command menu: buffer IS a leading /token. Prefix matches rank first,
      // substring matches follow — /und and /do both find /undo.
      if (buf.startsWith("/") && !buf.includes(" ")) {
        const q = buf.slice(1).toLowerCase();
        const starts = commands.filter((c) => c.name.slice(1).toLowerCase().startsWith(q));
        const contains = q
          ? commands.filter((c) => !starts.includes(c) && c.name.slice(1).toLowerCase().includes(q))
          : [];
        const items = [...starts, ...contains].map((c) => ({
          label: `${c.name}${c.hasArg ? " <…>" : ""}`,
          desc: c.desc,
          insert: c.hasArg ? `${c.name} ` : c.name,
          submit: !c.hasArg,
        }));
        return items.length ? { start: 0, items } : null;
      }
      return null;
    };

    // Rows ABOVE the cursor that belong to our drawing (the frame's top edge
    // plus any rows the input wrapped onto). Every redraw rewinds by this
    // before erasing — without it, a wrapped input leaves its first row
    // stranded and the whole block marches one row down per keystroke.
    let drawnAbove = 0;
    // What the last render actually drew, for the resize path: the terminal
    // REFLOWS these rows at the new width, so the rewind there must be
    // recomputed from content lengths, not taken from stale drawnAbove.
    let lastTopLen = 0;
    let lastInputLen = 0;

    const rewind = () => `${drawnAbove ? `\x1b[${drawnAbove}A` : ""}\r\x1b[J`;

    const render = (rewindStr?: string) => {
      // Honest width: never pretend the terminal is wider than it is — a
      // floor above the real column count makes every drawn row wrap and
      // shreds the box when the user zooms in. 8 only guards degenerate PTYs.
      const cols = Math.max(process.stdout.columns ?? 80, 8);
      const top = opts.frameTop?.();
      const { rows: inputRows, col } = cursorGeometry(promptLen + buf.length, cols);

      // Redraw the frame top every time: that makes the box resize-correct and
      // self-healing after anything above erases it (e.g. the ctrl+o replay).
      let out = rewindStr ?? rewind();
      if (top) out += `${top}\n`;
      out += `${prompt}${buf}`;
      lastTopLen = top ? visibleLen(top) : 0;
      lastInputLen = promptLen + buf.length;

      let below = 0; // rows drawn BELOW the input's last row
      const menu = activeMenu();
      if (menu) {
        const items = menu.items;
        if (menuIdx >= items.length) menuIdx = items.length - 1;
        if (menuIdx < 0) menuIdx = 0;
        // Sliding window keeps the selection visible.
        const start = Math.max(0, Math.min(menuIdx - (MENU_ROWS - 2), items.length - MENU_ROWS));
        const end = Math.min(items.length, start + MENU_ROWS);
        const rowWidth = Math.max(8, cols - 2); // never wider than the terminal
        if (start > 0) { out += `\n${DIM()}  ↑ ${start} more${R}`; below++; }
        for (let i = start; i < end; i++) {
          const it = items[i]!;
          const line = ` ${it.label}${it.desc ? `  ${it.desc}` : ""} `;
          out += `\n${i === menuIdx ? INV : DIM()}${line.slice(0, rowWidth)}${R}`;
          below++;
        }
        if (end < items.length) { out += `\n${DIM()}  ↓ ${items.length - end} more${R}`; below++; }
        out += `\n${DIM()}  ↑↓ move · Tab complete · Enter run · Esc close${R}`; below++;
      }
      // The status row is the frame's bottom edge, so draw it even with a menu
      // open — otherwise opening the menu visibly removes the box's bottom.
      if (opts.status) {
        const st = typeof opts.status === "function" ? opts.status() : opts.status;
        if (st) {
          out += `\n${st}`;
          below += 1 + cursorGeometry(stripAnsi(st).length, cols).rows;
        }
      }

      if (below) out += `\x1b[${below}A`;
      out += `\r${col ? `\x1b[${col}C` : ""}`;
      drawnAbove = inputRows + (top ? 1 : 0);
      process.stdout.write(out);
    };

    const done = (result: string | null) => {
      stdin.removeListener("keypress", onKey);
      process.stdout.removeListener("resize", onResize);
      if (resizeTimer) clearTimeout(resizeTimer);
      const top = opts.frameTop?.();
      process.stdout.write(`${rewind()}${top ? `${top}\n` : ""}${prompt}${result ?? ""}\n`);
      drawnAbove = 0;
      resolve(result);
    };

    // Zoom/resize: redraw at the new width without waiting for a keypress.
    // The terminal has REFLOWED our old rows, so the rewind is recomputed
    // from the recorded content lengths (reflowRewind), never from stale
    // drawnAbove — that mismatch is what multiplied the frame top on every
    // zoom step. Debounced: a zoom gesture fires a burst of resize events,
    // and one redraw at the final size beats ten flickering intermediates.
    let resizeTimer: ReturnType<typeof setTimeout> | undefined;
    const onResize = () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        resizeTimer = undefined;
        const cols = Math.max(process.stdout.columns ?? 80, 8);
        const up = Math.min(reflowRewind(lastTopLen ? [lastTopLen] : [], lastInputLen, cols), 400);
        render(`${up ? `\x1b[${up}A` : ""}\r\x1b[J`);
      }, 40);
    };

    const pick = (menu: { start: number; items: MenuItem[] }, viaEnter: boolean) => {
      const it = menu.items[Math.min(menuIdx, menu.items.length - 1)]!;
      buf = buf.slice(0, menu.start) + it.insert;
      menuIdx = 0;
      if (viaEnter && it.submit) done(buf.trim());
      else render();
    };

    const onKey = (str: string, key: any) => {
      if (!key) return;
      if (key.ctrl && key.name === "c") {
        if (buf) { buf = ""; menuIdx = 0; menuClosed = false; render(); }
        else done(null);
        return;
      }
      if (key.ctrl && key.name === "o") {
        // Erase our whole block (frame included) so the caller's replay starts
        // from a clean row; render() then rebuilds the frame around it.
        process.stdout.write(rewind());
        drawnAbove = 0;
        opts.onCtrlO?.();
        render();
        return;
      }
      if (key.ctrl || key.meta) return;

      const menu = activeMenu();
      switch (key.name) {
        case "return": case "enter":
          if (menu) pick(menu, true);
          else done(buf);
          return;
        case "tab":
          if (menu) pick(menu, false);
          return;
        case "backspace":
          buf = buf.slice(0, -1);
          menuClosed = false;
          render();
          return;
        case "escape":
          menuClosed = true;
          render();
          return;
        case "up":
          if (menu) { menuIdx = Math.max(0, menuIdx - 1); render(); }
          else if (history.length) {
            histIdx = Math.max(0, histIdx - 1);
            buf = history[histIdx] ?? "";
            render();
          }
          return;
        case "down":
          if (menu) { menuIdx = Math.min(menu.items.length - 1, menuIdx + 1); render(); }
          else if (histIdx < history.length) {
            histIdx++;
            buf = histIdx === history.length ? "" : history[histIdx] ?? "";
            render();
          }
          return;
      }
      if (typeof str === "string" && str.length > 0 && str >= " ") {
        buf += str;
        menuClosed = false;
        histIdx = history.length;
        render();
      }
    };

    stdin.on("keypress", onKey);
    process.stdout.on("resize", onResize);
    render();
  });
}
