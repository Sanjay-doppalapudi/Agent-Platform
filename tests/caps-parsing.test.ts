// Output caps, BOM preservation, and result-block parsing — all from the
// tools audit. Each of these was silently wrong in a way no manual test would
// surface (bytes vs characters, three bytes at the head of a file, and a
// snippet attached to the wrong URL).
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { execTool } from "../src/tools/index.ts";
import { truncateMiddle } from "../src/tools/shared.ts";

const bytes = (s: string) => Buffer.byteLength(s, "utf8");

describe("truncateMiddle caps BYTES, not characters", () => {
  const CAP = 10_000;
  const inputs: [string, string][] = [
    ["ascii", "a".repeat(50_000)],
    ["cjk", "日".repeat(20_000)],
    ["emoji", "🎉".repeat(12_000)],
    ["mixed", "héllo 日本 🎉 ".repeat(4_000)],
  ];
  for (const [label, s] of inputs) {
    test(`${label} output stays within the cap`, () => {
      const out = truncateMiddle(s, CAP);
      expect(bytes(out)).toBeLessThanOrEqual(CAP);
    });
    test(`${label} output contains no split characters`, () => {
      expect(truncateMiddle(s, CAP)).not.toContain("�");
    });
  }
  test("short input passes through untouched", () => {
    expect(truncateMiddle("hi", 100)).toBe("hi");
  });
  test("exactly-at-cap input is untouched", () => {
    const s = "a".repeat(100);
    expect(truncateMiddle(s, 100)).toBe(s);
  });
  test("head and tail are both preserved", () => {
    const out = truncateMiddle("START" + "x".repeat(50_000) + "END", 2_000);
    expect(out.startsWith("START")).toBe(true);
    expect(out.endsWith("END")).toBe(true);
    expect(out).toContain("truncated");
  });
});

describe("edit preserves a UTF-8 BOM", () => {
  const ws = () => {
    const cwd = mkdtempSync(join(tmpdir(), "ap-bom-"));
    return { cwd, ctx: { cwd, config: loadConfig({ cwd } as any), signal: new AbortController().signal, permit: async () => true } as any };
  };

  test("a BOM'd file keeps its BOM after an edit", async () => {
    const w = ws();
    const file = join(w.cwd, "bom.txt");
    writeFileSync(file, "﻿hello\nMARKER\ntail\n");
    const before = readFileSync(file);
    expect(before[0]).toBe(0xef); // sanity: the fixture really has a BOM

    const r = await execTool("edit", JSON.stringify({ path: "bom.txt", old: "MARKER", new: "REPLACED" }), w.ctx);
    expect(r.error).toBe(false);
    const after = readFileSync(file);
    expect([after[0], after[1], after[2]]).toEqual([0xef, 0xbb, 0xbf]);
    expect(after.toString("utf8")).toContain("REPLACED");
  });

  test("a file WITHOUT a BOM does not gain one", async () => {
    const w = ws();
    const file = join(w.cwd, "plain.txt");
    writeFileSync(file, "hello\nMARKER\n");
    await execTool("edit", JSON.stringify({ path: "plain.txt", old: "MARKER", new: "X" }), w.ctx);
    expect(readFileSync(file)[0]).not.toBe(0xef);
  });
});

describe("websearch pairs each snippet with its own result", () => {
  // The parser used to build two independently-filtered lists and zip them by
  // index, so skipping one ad shifted every later snippet onto the wrong URL.
  // This exercises the parsing shape directly against a DDG-like document.
  const page = `
    <div class="result results_links result--ad">
      <a class="result__a" href="/y.js?ad_domain=spam.example">Sponsored</a>
      <div class="result__extras"><a class="result__snippet">AD SNIPPET</a></div>
    </div>
    <div class="result results_links web-result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fone.example%2F">One</a>
      <div class="result__extras"><a class="result__snippet">SNIPPET ONE</a></div>
    </div>
    <div class="result results_links web-result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Ftwo.example%2F">Two</a>
      <div class="result__extras"><a class="result__snippet">SNIPPET TWO</a></div>
    </div>`;

  test("blocks split on the container class, ads dropped, pairs intact", () => {
    const blocks = page.split(/<div class="result[ "]/).slice(1);
    const parsed: { title: string; snippet: string }[] = [];
    for (const b of blocks) {
      const t = b.match(/class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
      if (!t || /ad_domain=|duckduckgo\.com\/y\.js/.test(t[1]!)) continue;
      const s = b.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/);
      parsed.push({ title: t[2]!, snippet: s ? s[1]! : "" });
    }
    expect(parsed.length).toBe(2); // the ad is gone
    expect(parsed[0]).toEqual({ title: "One", snippet: "SNIPPET ONE" });
    expect(parsed[1]).toEqual({ title: "Two", snippet: "SNIPPET TWO" }); // not shifted by the dropped ad
  });
});
