// Themes + input frame geometry (pure string work — no terminal needed).
import { describe, expect, test } from "bun:test";
import { currentTheme, cursorGeometry, EFFORT_LEVELS, fitSegments, frameBottom, frameTop, frameWidth, paint, parseEffort, reflowRewind, setTheme, THEMES, themeNames } from "../src/theme.ts";
import { MdRenderer } from "../src/md.ts";

describe("markdown nesting (regression)", () => {
  const render = (src: string) => new MdRenderer().push(src + "\n");

  test("bold survives an inline code span inside it", () => {
    const out = render("**use `foo` carefully**");
    // A blanket reset would end the bold at the code span; the closer must be
    // attribute-specific so "carefully" is still bold.
    expect(out).toContain("\x1b[1m"); // bold opened
    const afterCode = out.slice(out.indexOf("foo"));
    expect(afterCode).not.toContain("\x1b[0m\x1b[22m"); // no blanket reset before bold ends
    expect(out.indexOf("\x1b[22m")).toBeGreaterThan(out.indexOf("carefully"));
  });

  test("code spans reset only the foreground", () => {
    const out = render("plain `code` plain");
    expect(out).toContain("\x1b[39m");
    expect(out).not.toContain("\x1b[0m\x1b[39m");
  });

  test("mono theme strips answer-text color too", () => {
    setTheme("mono");
    const out = render("# heading\n- bullet with `code`");
    expect(out).not.toContain("\x1b[36m"); // no hardcoded cyan leaks through
    setTheme("default");
  });

  test("code fences still render and toggle", () => {
    const out = render("```ts\nconst x = 1;\n```");
    expect(out).toContain("╭─ ts");
    expect(out).toContain("╰─");
  });
});

describe("frameBottom color integrity (regression)", () => {
  // The bottom edge lost its color partway across: wrapping the WHOLE line in
  // a border color let a reset inside the status kill it, so the trailing
  // "───╯" rendered in the terminal default (looked white).
  const border = (s: string) => `\x1b[2;36m${s}\x1b[0m`;
  const status = `\x1b[2mmodel\x1b[0m \x1b[33mctx 62%\x1b[0m`;

  test("border color is re-applied after the status, not inherited through it", () => {
    const out = frameBottom(60, status, border);
    // The closing run must open AFTER the status (so the status's own resets
    // cannot kill it) and must not be reset again before the corner.
    const openedAt = out.lastIndexOf("\x1b[2;36m");
    expect(openedAt).toBeGreaterThan(out.indexOf("ctx 62%"));
    const closingRun = out.slice(openedAt, out.lastIndexOf("╯"));
    expect(closingRun).not.toContain("\x1b[0m");
  });

  test("the closing corner is inside a colored run", () => {
    const out = frameBottom(60, status, border);
    const lastOpen = out.lastIndexOf("\x1b[2;36m");
    const lastReset = out.lastIndexOf("\x1b[0m");
    expect(lastOpen).toBeLessThan(out.lastIndexOf("╯"));
    expect(lastReset).toBeGreaterThan(out.lastIndexOf("╯")); // reset comes after the corner
  });

  test("visible width is unaffected by the painter", () => {
    const plain = frameBottom(60, "model ctx 62%");
    const painted = frameBottom(60, "model ctx 62%", border);
    const vis = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
    expect(vis(painted)).toBe(vis(plain));
    expect(vis(painted).length).toBe(60);
  });

  test("empty status still paints the whole closed edge", () => {
    const out = frameBottom(20, "", border);
    expect(out.startsWith("\x1b[2;36m╰")).toBe(true);
    expect(out).toContain("╯\x1b[0m");
  });
});

describe("cursorGeometry (wrapped-input regression)", () => {
  // The input line wrapping is what actually shredded the box: the renderer
  // assumed prompt+buf was always exactly one row.
  test("short input stays on row 0", () => {
    expect(cursorGeometry(10, 80)).toEqual({ rows: 0, col: 10 });
  });
  test("deferred wrap: exactly cols chars stays on the same row", () => {
    // The terminal parks the cursor on the LAST column, it does not wrap yet.
    expect(cursorGeometry(80, 80)).toEqual({ rows: 0, col: 80 });
  });
  test("one past the width moves to the next row, column 1", () => {
    expect(cursorGeometry(81, 80)).toEqual({ rows: 1, col: 1 });
  });
  test("multiple wraps", () => {
    expect(cursorGeometry(200, 80)).toEqual({ rows: 2, col: 40 });
    expect(cursorGeometry(160, 80)).toEqual({ rows: 1, col: 80 });
    expect(cursorGeometry(161, 80)).toEqual({ rows: 2, col: 1 });
  });
  test("empty input emits no cursor movement", () => {
    expect(cursorGeometry(0, 80)).toEqual({ rows: 0, col: 0 });
  });
  test("degenerate widths never divide by zero", () => {
    expect(cursorGeometry(5, 0).rows).toBeGreaterThanOrEqual(0);
    expect(() => cursorGeometry(5, 1)).not.toThrow();
  });
});

describe("parseEffort", () => {
  test("canonical levels", () => {
    for (const l of EFFORT_LEVELS) expect(parseEffort(l)).toBe(l);
  });
  test("aliases people actually type", () => {
    expect(parseEffort("max")).toBe("high");     // reported by a user
    expect(parseEffort("MAX")).toBe("high");
    expect(parseEffort("med")).toBe("medium");
    expect(parseEffort("min")).toBe("minimal");
  });
  test("off/default/none all disable", () => {
    for (const a of ["off", "default", "none", "auto"]) expect(parseEffort(a)).toBe("off");
  });
  test("garbage is rejected, not silently coerced", () => {
    expect(parseEffort("turbo")).toBeNull();
    expect(parseEffort("")).toBeNull();
  });
});

const visible = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

describe("themes", () => {
  test("every theme defines every slot", () => {
    for (const [name, t] of Object.entries(THEMES)) {
      for (const k of ["accent", "text", "dim", "success", "error", "warn", "border", "add", "del"] as const) {
        expect(typeof t[k], `${name}.${k}`).toBe("string");
      }
      expect(t.name).toBe(name);
      expect(t.desc.length).toBeGreaterThan(0);
    }
  });
  test("switching works and unknown names are rejected", () => {
    expect(setTheme("nord")).toBe(true);
    expect(currentTheme().name).toBe("nord");
    expect(setTheme("chartreuse")).toBe(false);
    expect(currentTheme().name).toBe("nord"); // unchanged after a bad name
    setTheme("default");
  });
  test("mono emits no color codes", () => {
    setTheme("mono");
    expect(paint(currentTheme().accent, "hi")).toBe("hi");
    setTheme("default");
  });
  test("names are stable ids", () => expect(themeNames()).toContain("dracula"));
});

describe("frameWidth", () => {
  test("never exceeds the real terminal width (REGRESSION: the old floor of 28 wrapped every frame row on zoomed-in terminals)", () => {
    expect(frameWidth(10)).toBe(9); // was 28 — wider than the terminal itself
    expect(frameWidth(24)).toBe(23);
    expect(frameWidth(400)).toBe(160);
    expect(frameWidth(80)).toBe(79);
  });
});

describe("frameTop", () => {
  test("embeds the label and fills to width", () => {
    const t = frameTop(40, "code");
    expect(visible(t).length).toBe(40);
    expect(t.startsWith("╭─ code ")).toBe(true);
    expect(t.endsWith("╮")).toBe(true);
  });
  test("drops the label when too narrow", () => {
    const t = frameTop(10, "a-very-long-mode-name");
    expect(visible(t).length).toBe(10);
    expect(t).toBe("╭────────╮");
  });
  test("painted pieces: label color stays inside the label, border pieces are painted separately", () => {
    const border = (s: string) => `<B>${s}</B>`;
    const label = (s: string) => `<L>${s}</L>`;
    const t = frameTop(40, "◆ code", border, label);
    expect(t).toBe(`<B>╭─ </B><L>◆ code</L><B> ${"─".repeat(29)}╮</B>`);
  });
});

describe("reflowRewind", () => {
  // The multiplying-box bug: after a zoom, the terminal REWRAPS previously
  // drawn rows, so the rewind distance must be recomputed from content
  // lengths at the new width — stale row counts land mid-block.
  test("no reflow needed: wide terminal, one row above", () => {
    expect(reflowRewind([59], 5, 80)).toBe(1);
  });
  test("shrunk terminal: the 59-char top edge now occupies two rows", () => {
    expect(reflowRewind([59], 5, 36)).toBe(2);
  });
  test("deferred wrap: an exactly-cols line is still ONE row", () => {
    expect(reflowRewind([36], 5, 36)).toBe(1);
  });
  test("cursor line itself wraps too", () => {
    expect(reflowRewind([59], 40, 36)).toBe(3); // top→2 rows + 1 wrapped input row
  });
  test("nothing above, short cursor line → no rewind", () => {
    expect(reflowRewind([], 10, 36)).toBe(0);
  });
  test("multiple lines above (picker block)", () => {
    // title 30, filter 10, three items 20 each at 25 cols: 30→2 rows, rest 1 each.
    expect(reflowRewind([30, 10, 20, 20], 20, 25)).toBe(5);
  });
});

describe("fitSegments", () => {
  const seg = (text: string, prio: number) => ({ text, prio });
  test("everything fits → everything stays, in display order", () => {
    expect(fitSegments([seg("a", 0), seg("b", 3), seg("c", 0)], 80, " · ")).toBe("a · b · c");
  });
  test("under pressure the highest prio drops first", () => {
    const s = [seg("model", 0), seg("effort high", 3), seg("ctx 4%", 0), seg("~$0.03", 4)];
    expect(fitSegments(s, 100, " · ")).toBe("model · effort high · ctx 4% · ~$0.03");
    expect(fitSegments(s, 30, " · ")).toBe("model · effort high · ctx 4%"); // cost (prio 4) dropped
    expect(fitSegments(s, 20, " · ")).toBe("model · ctx 4%"); // then effort (prio 3)
  });
  test("ties drop the rightmost segment first", () => {
    const s = [seg("left", 2), seg("mid", 0), seg("right", 2)];
    expect(fitSegments(s, 11, " · ")).toBe("left · mid");
  });
  test("prio-0 segments never drop, even when still over budget", () => {
    expect(fitSegments([seg("a-long-model-name", 0), seg("ctx 4%", 0)], 5, " · ")).toBe("a-long-model-name · ctx 4%");
  });
  test("ANSI codes are invisible to the width math", () => {
    const s = [seg("\x1b[2mmodel\x1b[0m", 0), seg("\x1b[33mctx 90%\x1b[0m", 0), seg("\x1b[2m~$1\x1b[0m", 4)];
    expect(fitSegments(s, 14, " · ")).toBe("\x1b[2mmodel\x1b[0m · \x1b[33mctx 90%\x1b[0m");
  });
});

describe("frameBottom", () => {
  test("carries status text at exact width", () => {
    const b = frameBottom(50, "model · ctx 4%");
    expect(visible(b).length).toBe(50);
    expect(b).toContain("ctx 4%");
    expect(b.endsWith("╯")).toBe(true);
  });
  test("truncates over-long status instead of wrapping", () => {
    const b = frameBottom(30, "x".repeat(200));
    expect(visible(b).length).toBeLessThanOrEqual(30);
    expect(b).toContain("…");
  });
  test("ANSI in the status does not corrupt width math", () => {
    const colored = `\x1b[36mmodel\x1b[0m · \x1b[2mctx 9%\x1b[0m`;
    const b = frameBottom(44, colored);
    expect(visible(b).length).toBe(44);
  });
  test("empty status → plain edge", () => {
    expect(frameBottom(12)).toBe("╰──────────╯");
  });
});
