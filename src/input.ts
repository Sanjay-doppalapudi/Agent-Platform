// Minimal raw-mode line reader with a slash-command menu. Zero dependencies.
// Typing "/" opens the menu; typing filters it; ↑/↓ navigate; Enter/Tab select;
// Esc closes. ↑/↓ recall history when the menu is closed.
// Caller must have run readline.emitKeypressEvents(process.stdin) once.

const R = "\x1b[0m";
const DIM = "\x1b[2m";
const INV = "\x1b[7m";

export interface SlashCommand {
  name: string; // "/plan"
  desc: string;
  hasArg?: boolean;
}

const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

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

    const matches = (): SlashCommand[] =>
      !menuClosed && buf.startsWith("/") && !buf.includes(" ")
        ? commands.filter((c) => c.name.startsWith(buf))
        : [];

    const render = () => {
      let out = `\r\x1b[J${prompt}${buf}`;
      const m = matches();
      if (m.length) {
        if (menuIdx >= m.length) menuIdx = m.length - 1;
        const rows = Math.min(m.length, 8);
        for (let i = 0; i < rows; i++) {
          const c = m[i]!;
          const line = ` ${c.name}${c.hasArg ? " <…>" : ""}  ${c.desc} `;
          out += `\n${i === menuIdx ? INV : DIM}${line}${R}`;
        }
        out += `\x1b[${rows}A\r\x1b[${promptLen + buf.length}C`;
      }
      process.stdout.write(out);
    };

    const done = (result: string | null) => {
      stdin.removeListener("keypress", onKey);
      process.stdout.write(`\r\x1b[J${prompt}${result ?? ""}\n`);
      resolve(result);
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

      const m = matches();
      switch (key.name) {
        case "return": case "enter": {
          if (m.length) {
            const sel = m[Math.min(menuIdx, m.length - 1)]!;
            if (sel.hasArg) { buf = sel.name + " "; menuIdx = 0; render(); }
            else done(sel.name);
          } else {
            done(buf);
          }
          return;
        }
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
          if (m.length) { menuIdx = (menuIdx + m.length - 1) % m.length; render(); }
          else if (history.length) {
            histIdx = Math.max(0, histIdx - 1);
            buf = history[histIdx] ?? "";
            render();
          }
          return;
        case "down":
          if (m.length) { menuIdx = (menuIdx + 1) % m.length; render(); }
          else if (histIdx < history.length) {
            histIdx++;
            buf = histIdx === history.length ? "" : history[histIdx] ?? "";
            render();
          }
          return;
        case "tab": {
          if (m.length) {
            const sel = m[Math.min(menuIdx, m.length - 1)]!;
            buf = sel.name + (sel.hasArg ? " " : "");
            menuIdx = 0;
            render();
          }
          return;
        }
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
