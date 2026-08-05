// Themes + input frame geometry (pure string work — no terminal needed).
import { describe, expect, test } from "bun:test";
import { currentTheme, EFFORT_LEVELS, frameBottom, frameTop, frameWidth, paint, parseEffort, setTheme, THEMES, themeNames } from "../src/theme.ts";

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
  test("clamps absurd terminal sizes", () => {
    expect(frameWidth(10)).toBe(28);
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
