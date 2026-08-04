// Minimal raw-mode line reader with dropdown menus. Zero dependencies.
// "/" opens the command menu; "@" opens the file picker (lazy rg --files).
// Typing filters; ↑/↓ scroll (windowed); Enter/Tab select; Esc closes.
// ↑/↓ recall history when no menu is open.
// Caller must have run readline.emitKeypressEvents(process.stdin) once.

const R = "\x1b[0m";
const DIM = "\x1b[2m";
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

    const render = () => {
      let out = `\r\x1b[J${prompt}${buf}`;
      const menu = activeMenu();
      if (menu) {
        const items = menu.items;
        if (menuIdx >= items.length) menuIdx = items.length - 1;
        if (menuIdx < 0) menuIdx = 0;
        // Sliding window keeps the selection visible.
        const start = Math.max(0, Math.min(menuIdx - (MENU_ROWS - 2), items.length - MENU_ROWS));
        const end = Math.min(items.length, start + MENU_ROWS);
        let rows = 0;
        if (start > 0) { out += `\n${DIM}  ↑ ${start} more${R}`; rows++; }
        for (let i = start; i < end; i++) {
          const it = items[i]!;
          const line = ` ${it.label}${it.desc ? `  ${it.desc}` : ""} `;
          out += `\n${i === menuIdx ? INV : DIM}${line.slice(0, Math.max(20, (process.stdout.columns ?? 80) - 2))}${R}`;
          rows++;
        }
        if (end < items.length) { out += `\n${DIM}  ↓ ${items.length - end} more${R}`; rows++; }
        out += `\n${DIM}  ↑↓ move · Tab complete · Enter run · Esc close${R}`; rows++;
        out += `\x1b[${rows}A\r\x1b[${promptLen + buf.length}C`;
      }
      process.stdout.write(out);
    };

    const done = (result: string | null) => {
      stdin.removeListener("keypress", onKey);
      process.stdout.write(`\r\x1b[J${prompt}${result ?? ""}\n`);
      resolve(result);
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
        process.stdout.write("\r\x1b[J");
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
    render();
  });
}
