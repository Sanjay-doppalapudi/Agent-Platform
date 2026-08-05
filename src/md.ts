// Line-buffered streaming markdown → ANSI renderer. Deltas are buffered
// until a full line exists, then rendered — correct fence/header handling
// with near-zero latency cost (lines complete every few tokens).

import { currentTheme } from "./theme.ts";

// Closers are ATTRIBUTE-SPECIFIC (SGR 22/23/39), never a blanket reset (SGR 0).
// A blanket reset inside a nested span kills the enclosing style too — that is
// why `**bold with `code` inside**` used to lose its bold after the code span.
const R = "\x1b[0m";
const OFF_INTENSITY = "\x1b[22m"; // ends bold AND dim
const OFF_ITALIC = "\x1b[23m";
const OFF_FG = "\x1b[39m"; // back to default foreground, keeps bold/italic
const BOLD = "\x1b[1m";
const ITAL = "\x1b[3m";
// Colors follow the active theme, so /theme and NO_COLOR reach answer text too.
const DIM = () => currentTheme().dim;
const CYAN = () => currentTheme().accent;
const YELLOW = () => currentTheme().warn;
const GREEN = () => currentTheme().add;

function width(): number {
  return Math.min(process.stdout.columns ?? 80, 100);
}

export class MdRenderer {
  private buf = "";
  private inCode = false;

  push(delta: string): string {
    this.buf += delta;
    let out = "";
    let nl: number;
    while ((nl = this.buf.indexOf("\n")) !== -1) {
      const line = this.buf.slice(0, nl);
      this.buf = this.buf.slice(nl + 1);
      out += this.renderLine(line) + "\n";
    }
    return out;
  }

  /** Render whatever is left (partial last line) and reset. */
  flush(): string {
    const rest = this.buf;
    this.buf = "";
    this.inCode = false;
    return rest ? this.renderLine(rest) : "";
  }

  private renderLine(line: string): string {
    const fence = line.match(/^\s*```\s*(\S*)\s*$/);
    if (fence) {
      this.inCode = !this.inCode;
      return this.inCode
        ? `${DIM()}╭─ ${fence[1] || "code"}${R}`
        : `${DIM()}╰─${R}`;
    }
    if (this.inCode) return `${DIM()}│${R} ${GREEN()}${line}${R}`;

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) return `${BOLD}${YELLOW()}${h[2]}${R}`;

    if (/^\s*([-_*])\1{2,}\s*$/.test(line)) return `${DIM()}${"─".repeat(width())}${R}`;

    if (/^\s*>\s?/.test(line)) return `${DIM()}${ITAL}${line.replace(/^\s*>\s?/, "▌ ")}${R}`;

    return this.inline(line);
  }

  private inline(s: string): string {
    // Order matters with attribute-specific closers: emphasis wrappers survive
    // a nested code span because the span only resets the FOREGROUND.
    s = s.replace(/`([^`]+)`/g, `${CYAN()}$1${OFF_FG}`);
    s = s.replace(/\*\*([^*]+)\*\*/g, `${BOLD}$1${OFF_INTENSITY}`);
    s = s.replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,!?:;]|$)/g, `$1${ITAL}$2${OFF_ITALIC}`);
    s = s.replace(/^(\s*)(?:[-*+]|(\d+\.))\s/, (_m, sp: string, num?: string) =>
      `${sp}${CYAN()}${num ?? "•"}${OFF_FG} `);
    return s;
  }
}
