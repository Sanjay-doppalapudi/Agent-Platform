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

/**
 * Usable frame width: terminal columns minus a one-column safety margin,
 * capped at 160. There is NO lower clamp above the terminal's real width —
 * a floor (this used to be 28) draws lines wider than a zoomed-in terminal,
 * and every frame row then wraps, shredding the box. Honest width always
 * beats a pretty minimum.
 */
export function frameWidth(cols = process.stdout.columns ?? 80): number {
  return Math.max(4, Math.min(cols - 1, 160));
}

export const visibleLen = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "").length;

/**
 * Truncate to `width` VISIBLE characters, keeping ANSI sequences intact so a
 * cut can never land inside an escape (which would leak raw bytes or a stuck
 * colour). Shared by the live status line and the watch viewer.
 */
export function fitLine(s: string, width: number): string {
  if (visibleLen(s) <= width) return s;
  let out = "";
  let seen = 0;
  for (let i = 0; i < s.length && seen < width - 1; i++) {
    const ch = s[i]!;
    if (ch === "\x1b") {
      const end = s.indexOf("m", i);
      if (end !== -1) { out += s.slice(i, end + 1); i = end; continue; }
    }
    out += ch;
    seen++;
  }
  return out + "…";
}

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
 * Physical rows from the top of a previously drawn block to the cursor AFTER
 * the terminal REFLOWED it at `cols`. Windows Terminal (and iTerm etc.)
 * rewrap existing rows on resize, so a rewind computed from the pre-resize
 * geometry lands mid-block — each zoom step then paints a fresh frame top
 * next to the stranded old one (the "multiplying box"). The fix: we know the
 * visible length of every line we drew, so the post-reflow geometry is fully
 * computable. `above` = visible lengths of the full lines above the cursor's
 * line; `cursorLen` = characters before the cursor on its own logical line.
 * Deferred wrap applies throughout (an exactly-cols line is still one row).
 */
export function reflowRewind(above: number[], cursorLen: number, cols: number): number {
  let rows = 0;
  for (const len of above) rows += cursorGeometry(len, cols).rows + 1;
  return rows + cursorGeometry(cursorLen, cols).rows;
}

/**
 * Top edge: `╭─ label ─────────╮`. With no painters this returns a plain
 * string (the caller may paint the whole line). With `border`/`labelPaint`
 * the pieces are painted individually — same discipline as frameBottom, so a
 * label color can never leak into the fill and a reset inside the label can
 * never kill the border color. Label is dropped when the terminal is too
 * narrow to hold it.
 */
export function frameTop(
  width: number,
  label = "",
  border: (s: string) => string = (s) => s,
  labelPaint?: (s: string) => string,
): string {
  const w = Math.max(4, width);
  const len = visibleLen(label);
  if (label && w >= len + 8) {
    const fill = "─".repeat(Math.max(1, w - len - 5));
    return `${border("╭─ ")}${(labelPaint ?? border)(label)}${border(` ${fill}╮`)}`;
  }
  return border(`╭${"─".repeat(w - 2)}╮`);
}

export interface StatusSegment {
  text: string;
  /** Drop order under pressure: higher drops first; 0 never drops. */
  prio: number;
}

/**
 * Join status segments with `sep`, dropping the highest-prio segments
 * (rightmost first on ties) until the visible length fits `budget`. Display
 * order is preserved. Prio-0 segments survive even if the result still
 * overflows — frameBottom's ANSI-safe truncation is the backstop.
 */
export function fitSegments(segments: StatusSegment[], budget: number, sep: string): string {
  const alive = segments.slice();
  const join = () => alive.map((s) => s.text).join(sep);
  while (alive.length > 1 && visibleLen(join()) > budget) {
    let worst = -1;
    let worstPrio = 0;
    for (let i = 0; i < alive.length; i++) {
      const p = alive[i]!.prio;
      if (p > 0 && p >= worstPrio) { worst = i; worstPrio = p; }
    }
    if (worst === -1) break;
    alive.splice(worst, 1);
  }
  return join();
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
