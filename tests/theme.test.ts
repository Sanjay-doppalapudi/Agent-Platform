// Themes + input frame geometry (pure string work — no terminal needed).
import { describe, expect, test } from "bun:test";
import { currentTheme, frameBottom, frameTop, frameWidth, paint, setTheme, THEMES, themeNames } from "../src/theme.ts";

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
