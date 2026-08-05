// Terminal themes + the input frame. Pure string work, no dependencies, no
// terminal queries — a theme is just a palette of ANSI codes, and the frame is
// built from the known column count. Colors auto-disable when NO_COLOR is set
// or stdout isn't a TTY (piped output stays clean for scripts).

export interface Theme {
  name: string;
  desc: string;
  accent: string;   // prompt caret, tool actions, highlights
  text: string;     // normal emphasis (headings, labels)
  dim: string;      // metadata, reasoning, hints
  success: string;
  error: string;
  warn: string;
  border: string;   // the input frame
  add: string;      // diff +
  del: string;      // diff -
}

const RESET = "\x1b[0m";
const c256 = (n: number) => `\x1b[38;5;${n}m`;

export const THEMES: Record<string, Theme> = {
  default: {
    name: "default", desc: "AP cyan on your terminal's own palette",
    accent: "\x1b[36m", text: "\x1b[1m", dim: "\x1b[2m",
    success: "\x1b[32m", error: "\x1b[31m", warn: "\x1b[33m",
    border: "\x1b[2;36m", add: "\x1b[32m", del: "\x1b[31m",
  },
  mono: {
    name: "mono", desc: "no color — plain text (also used when NO_COLOR is set)",
    accent: "", text: "\x1b[1m", dim: "\x1b[2m",
    success: "", error: "", warn: "", border: "\x1b[2m", add: "", del: "",
  },
  nord: {
    name: "nord", desc: "cool arctic blues",
    accent: c256(110), text: c256(189), dim: c256(103),
    success: c256(108), error: c256(174), warn: c256(179),
    border: c256(60), add: c256(108), del: c256(174),
  },
  dracula: {
    name: "dracula", desc: "purple/pink on dark",
    accent: c256(141), text: c256(189), dim: c256(103),
    success: c256(84), error: c256(212), warn: c256(228),
    border: c256(97), add: c256(84), del: c256(212),
  },
  gruvbox: {
    name: "gruvbox", desc: "warm retro earth tones",
    accent: c256(214), text: c256(223), dim: c256(245),
    success: c256(142), error: c256(167), warn: c256(214),
    border: c256(101), add: c256(142), del: c256(167),
  },
  solarized: {
    name: "solarized", desc: "muted blue/cyan classic",
    accent: c256(37), text: c256(244), dim: c256(240),
    success: c256(64), error: c256(160), warn: c256(136),
    border: c256(240), add: c256(64), del: c256(160),
  },
  matrix: {
    name: "matrix", desc: "all green, all the time",
    accent: c256(46), text: c256(120), dim: c256(28),
    success: c256(46), error: c256(196), warn: c256(226),
    border: c256(22), add: c256(46), del: c256(196),
  },
};

export function themeNames(): string[] {
  return Object.keys(THEMES);
}

/** Reasoning-effort levels AP accepts, plus the aliases people actually type. */
export const EFFORT_LEVELS = ["minimal", "low", "medium", "high"] as const;
export type EffortLevel = (typeof EFFORT_LEVELS)[number];
const EFFORT_ALIASES: Record<string, EffortLevel | "off"> = {
  max: "high", maximum: "high", hi: "high", full: "high",
  med: "medium", mid: "medium", normal: "medium",
  min: "minimal", lowest: "minimal", none: "off", default: "off", auto: "off",
};

/** Normalize an /effort argument → a level, "off", or null when unusable. */
export function parseEffort(arg: string): EffortLevel | "off" | null {
  const a = arg.trim().toLowerCase();
  if (!a) return null;
  if (a === "off") return "off";
  if ((EFFORT_LEVELS as readonly string[]).includes(a)) return a as EffortLevel;
  return EFFORT_ALIASES[a] ?? null;
}

const colorless = !!process.env.NO_COLOR || process.env.TERM === "dumb";
let active: Theme = colorless ? THEMES["mono"]! : THEMES["default"]!;

export function currentTheme(): Theme {
  return active;
}

/** Switch themes; returns false for an unknown name (caller lists valid ones). */
export function setTheme(name: string): boolean {
  const t = THEMES[name.trim().toLowerCase()];
  if (!t) return false;
  active = colorless && t.name !== "mono" ? { ...t, ...THEMES["mono"]!, name: t.name } : t;
  return true;
}

/** Wrap text in a theme color (no-ops when the color is empty). */
export function paint(code: string, s: string): string {
  return code ? `${code}${s}${RESET}` : s;
}

// --- input frame -----------------------------------------------------------

/** Usable frame width: terminal columns, clamped to something sane. */
export function frameWidth(cols = process.stdout.columns ?? 80): number {
  return Math.max(28, Math.min(cols - 1, 160));
}

export const visibleLen = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "").length;

/**
 * Where the cursor ends up after printing `len` visible chars from column 0:
 * `rows` = terminal rows consumed BELOW the first one, `col` = 1-based column
 * on that last row (0 when nothing was printed).
 *
 * Deferred wrap matters here: after exactly `cols` characters the cursor sits
 * on the last column of the SAME row, not column 0 of the next one — so the
 * math is `(len - 1) / cols`, never `len / cols`.
 */
export function cursorGeometry(len: number, cols: number): { rows: number; col: number } {
  const c = Math.max(1, cols);
  const rows = Math.floor(Math.max(0, len - 1) / c);
  return { rows, col: len - rows * c };
}

/**
 * Top edge: `╭─ label ─────────╮`. Plain string — the caller paints it.
 * Label is dropped entirely when the terminal is too narrow to hold it.
 */
export function frameTop(width: number, label = ""): string {
  const w = Math.max(4, width);
  if (label && w >= label.length + 8) {
    const fill = "─".repeat(Math.max(1, w - label.length - 5));
    return `╭─ ${label} ${fill}╮`;
  }
  return `╭${"─".repeat(w - 2)}╮`;
}

/**
 * Bottom edge carrying the status text: `╰─ model · ctx 4% ───╯`.
 *
 * The status usually contains its own SGR codes, and any reset inside it
 * (`\x1b[0m`) would terminate a color wrapped around the WHOLE line — leaving
 * the trailing `───╯` in the terminal's default color. So the border pieces
 * are painted individually and the status is passed through untouched.
 * Width math uses visible (ANSI-stripped) length; an over-long status is
 * truncated rather than wrapped, because wrapping breaks the box.
 */
export function frameBottom(width: number, status = "", border: (s: string) => string = (s) => s): string {
  const w = Math.max(4, width);
  if (!status) return border(`╰${"─".repeat(w - 2)}╯`);
  let s = status;
  const budget = w - 6;
  if (visibleLen(s) > budget) {
    // Trim to budget while keeping ANSI intact: cut on visible characters.
    let out = "";
    let seen = 0;
    for (let i = 0; i < s.length && seen < budget - 1; i++) {
      const ch = s[i]!;
      if (ch === "\x1b") {
        const end = s.indexOf("m", i);
        if (end !== -1) { out += s.slice(i, end + 1); i = end; continue; }
      }
      out += ch;
      seen++;
    }
    s = out + "…";
  }
  const fill = "─".repeat(Math.max(1, w - visibleLen(s) - 5));
  return `${border("╰─ ")}${s}${border(` ${fill}╯`)}`;
}
