// A tiny virtual terminal, used to replay real rendering scenarios against
// readLine(). This exists because the input renderer's cursor math has broken
// three separate ways (wrapped input, ctrl+o, menus) and none of it is
// observable from unit tests on pure functions — you have to model the screen.
import { describe, expect, test } from "bun:test";
import { emitKeypressEvents } from "node:readline";
import { readLine } from "../src/input.ts";
import { frameBottom, frameTop, setTheme } from "../src/theme.ts";

/** Minimal VT: printable chars, CR/LF, CUU/CUD/CUF/CUB, ED(0/2), EL, SGR. */
class VT {
  rows: string[][] = [[]];
  cy = 0;
  cx = 0;
  private pendingWrap = false;

  constructor(public cols: number) {}

  private row(y: number): string[] {
    while (this.rows.length <= y) this.rows.push([]);
    return this.rows[y]!;
  }

  private put(ch: string) {
    if (this.pendingWrap) { this.cx = 0; this.cy++; this.pendingWrap = false; }
    const r = this.row(this.cy);
    while (r.length < this.cx) r.push(" ");
    r[this.cx] = ch;
    if (this.cx + 1 >= this.cols) this.pendingWrap = true; // deferred wrap
    else this.cx++;
  }

  write(s: string) {
    for (let i = 0; i < s.length; i++) {
      const ch = s[i]!;
      if (ch === "\x1b" && s[i + 1] === "[") {
        const m = /^\x1b\[([0-9;]*)([A-Za-z])/.exec(s.slice(i));
        if (m) {
          const n = m[1] === "" ? 1 : parseInt(m[1]!.split(";")[0]!, 10) || (m[2] === "J" || m[2] === "K" ? 0 : 1);
          switch (m[2]) {
            case "A": this.cy = Math.max(0, this.cy - n); this.pendingWrap = false; break;
            case "B": this.cy += n; this.pendingWrap = false; break;
            case "C": this.cx = Math.min(this.cols - 1, this.cx + n); this.pendingWrap = false; break;
            case "D": this.cx = Math.max(0, this.cx - n); this.pendingWrap = false; break;
            case "H": this.cy = 0; this.cx = 0; this.pendingWrap = false; break;
            case "J":
              if (n === 2) { this.rows = [[]]; this.cy = 0; this.cx = 0; }
              else { this.row(this.cy).length = Math.min(this.row(this.cy).length, this.cx); this.rows.length = this.cy + 1; }
              this.pendingWrap = false;
              break;
            case "K": this.row(this.cy).length = Math.min(this.row(this.cy).length, this.cx); break;
            default: break; // SGR and friends: no geometry effect
          }
          i += m[0].length - 1;
          continue;
        }
      }
      if (ch === "\r") { this.cx = 0; this.pendingWrap = false; continue; }
      if (ch === "\n") { this.cy++; this.cx = 0; this.pendingWrap = false; this.row(this.cy); continue; }
      this.put(ch);
    }
  }

  screen(): string[] {
    return this.rows.map((r) => r.join("").replace(/\s+$/, ""));
  }
}

/** Drive readLine against a VT, feeding keystrokes, and return the screen. */
async function drive(opts: {
  cols: number;
  keys: string[];
  framed?: boolean;
  submit?: boolean;
}): Promise<{ screen: string[]; result: string | null }> {
  const vt = new VT(opts.cols);
  const realWrite = process.stdout.write.bind(process.stdout);
  const realCols = process.stdout.columns;
  (process.stdout as any).write = (s: any) => { vt.write(String(s)); return true; };
  Object.defineProperty(process.stdout, "columns", { value: opts.cols, configurable: true });
  emitKeypressEvents(process.stdin);

  const status = () => frameBottom(opts.cols - 1, "model · ctx 4%", (s) => s);
  const p = readLine({
    prompt: opts.framed ? "│ › " : "code › ",
    commands: [{ name: "/plan", desc: "plan mode" }, { name: "/code", desc: "code mode" }],
    history: [],
    frameTop: opts.framed ? () => frameTop(opts.cols - 1, "code") : undefined,
    status,
  });

  for (const k of opts.keys) process.stdin.emit("keypress", k, { name: k, sequence: k });

  let result: string | null = null;
  let screen: string[];
  if (opts.submit) {
    process.stdin.emit("keypress", "\r", { name: "return" });
    result = await p;
    screen = vt.screen();
  } else {
    screen = vt.screen(); // snapshot the live frame BEFORE tearing down
    // Every readLine must be resolved, or its keypress listener survives and
    // draws into the NEXT test's terminal (two ctrl+c: clear buffer, then EOF).
    process.stdin.emit("keypress", "\x03", { name: "c", ctrl: true });
    process.stdin.emit("keypress", "\x03", { name: "c", ctrl: true });
    result = await p;
  }

  (process.stdout as any).write = realWrite;
  Object.defineProperty(process.stdout, "columns", { value: realCols, configurable: true });
  return { screen, result };
}

setTheme("mono"); // keep the VT free of SGR noise

describe("framed input rendering", () => {
  test("short input: exactly one box, one prompt row", async () => {
    const { screen } = await drive({ cols: 60, framed: true, keys: ["h", "i"] });
    const promptRows = screen.filter((r) => r.includes("› "));
    expect(promptRows.length).toBe(1);
    expect(screen[0]!.startsWith("╭─ code")).toBe(true);
    expect(screen.some((r) => r.startsWith("╰─"))).toBe(true);
    expect(promptRows[0]).toContain("hi");
  });

  test("WRAPPED input does not duplicate the prompt row (the shredded-box bug)", async () => {
    // 60 chars of input in a 40-column terminal → the input wraps twice.
    const keys = Array.from({ length: 60 }, (_, i) => String.fromCharCode(97 + (i % 26)));
    const { screen } = await drive({ cols: 40, framed: true, keys });
    const promptRows = screen.filter((r) => r.includes("› "));
    expect(promptRows.length).toBe(1); // was: one stale copy per keystroke
    const tops = screen.filter((r) => r.startsWith("╭─"));
    const bottoms = screen.filter((r) => r.startsWith("╰─"));
    expect(tops.length).toBe(1);
    expect(bottoms.length).toBe(1);
    // The box must still be in the right order: top, input, …, bottom.
    expect(screen.indexOf(tops[0]!)).toBeLessThan(screen.indexOf(promptRows[0]!));
    expect(screen.indexOf(bottoms[0]!)).toBeGreaterThan(screen.indexOf(promptRows[0]!));
  });

  test("wrapped input keeps every typed character on screen", async () => {
    const text = "the quick brown fox jumps over the lazy dog and keeps running";
    const { screen } = await drive({ cols: 40, framed: true, keys: text.split("") });
    const body = screen.join("");
    expect(body).toContain("the quick brown");
    expect(body).toContain("keeps running");
  });

  test("submitting a wrapped line leaves no duplicate rows", async () => {
    const keys = Array.from({ length: 55 }, () => "x");
    const { screen, result } = await drive({ cols: 40, framed: true, keys, submit: true });
    expect(result).toBe("x".repeat(55));
    expect(screen.filter((r) => r.includes("› ")).length).toBe(1);
    expect(screen.filter((r) => r.startsWith("╭─")).length).toBe(1);
  });

  test("menu open keeps the bottom edge (box never loses its floor)", async () => {
    const { screen } = await drive({ cols: 60, framed: true, keys: ["/"] });
    expect(screen.some((r) => r.startsWith("╰─"))).toBe(true);
    expect(screen.some((r) => r.includes("/plan"))).toBe(true);
    expect(screen.filter((r) => r.startsWith("╭─")).length).toBe(1);
  });

  test("menu rows never exceed the terminal width", async () => {
    const { screen } = await drive({ cols: 30, framed: true, keys: ["/"] });
    for (const row of screen) expect(row.length).toBeLessThanOrEqual(30);
  });

  test("unframed mode still renders a single prompt row", async () => {
    const { screen } = await drive({ cols: 50, keys: ["h", "i"] });
    expect(screen.filter((r) => r.includes("› ")).length).toBe(1);
    expect(screen.some((r) => r.startsWith("╭─"))).toBe(false);
  });
});
