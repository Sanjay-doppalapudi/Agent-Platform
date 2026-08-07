// Artifact tool: slug discipline, containment, CSP injection, size cap.
// The slug is the only model-controlled path component — these tests are the
// two locks (alphabet + containment) plus the no-network guarantee.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { artifactTool, slugify, withCsp } from "../src/tools/artifact.ts";

function ctxFor(dataDir: string) {
  return { cwd: dataDir, config: { dataDir } as any, signal: new AbortController().signal, permit: async () => true } as any;
}

describe("slugify", () => {
  test("titles become filesystem-safe slugs", () => {
    expect(slugify("Q3 Revenue — Report!")).toBe("q3-revenue-report");
    expect(slugify("///")).toBe("artifact");
    expect(slugify("x".repeat(200)).length).toBeLessThanOrEqual(60);
  });
});

describe("withCsp", () => {
  test("injects after <head> when present", () => {
    const out = withCsp("<html><head><title>t</title></head><body>x</body></html>");
    expect(out).toContain("Content-Security-Policy");
    expect(out.indexOf("Content-Security-Policy")).toBeLessThan(out.indexOf("<title>"));
  });
  test("wraps headless fragments into a full page", () => {
    const out = withCsp("<h1>hi</h1>");
    expect(out).toContain("<!doctype html>");
    expect(out).toContain("Content-Security-Policy");
    expect(out).toContain("<h1>hi</h1>");
  });
  test("the CSP blocks all network directions", () => {
    expect(withCsp("<p>x</p>")).toContain("default-src 'none'");
  });
});

describe("artifactTool", () => {
  test("writes a timestamped file and reports the path", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-artifact-"));
    try {
      const out = await artifactTool({ title: "My Report", html: "<h1>report</h1>" }, ctxFor(dir));
      expect(out).toContain("artifact saved:");
      const files = readdirSync(join(dir, "artifacts"));
      expect(files.length).toBe(1);
      expect(files[0]).toMatch(/^\d{8}-\d{6}-my-report\.html$/);
      const body = readFileSync(join(dir, "artifacts", files[0]!), "utf8");
      expect(body).toContain("Content-Security-Policy");
      expect(body).toContain("<h1>report</h1>");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("traversal-shaped slugs are rejected by the alphabet check", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-artifact2-"));
    try {
      for (const slug of ["../evil", "..\\evil", "a/b", "a\\b", ".hidden", "UPPER"]) {
        await expect(artifactTool({ title: "t", html: "<p>x</p>", slug }, ctxFor(dir))).rejects.toThrow();
      }
      // Empty slug is ABSENT, not invalid — it derives from the title.
      await expect(artifactTool({ title: "t", html: "<p>x</p>", slug: "" }, ctxFor(dir))).resolves.toContain("artifact saved");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("the size cap rejects oversized pages with a helpful message", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-artifact3-"));
    try {
      await expect(
        artifactTool({ title: "big", html: "x".repeat(2 * 1024 * 1024 + 1) }, ctxFor(dir)),
      ).rejects.toThrow(/cap/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("missing fields fail fast", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-artifact4-"));
    try {
      await expect(artifactTool({ title: "", html: "<p>x</p>" }, ctxFor(dir))).rejects.toThrow(/title/);
      await expect(artifactTool({ title: "t", html: "" }, ctxFor(dir))).rejects.toThrow(/html/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});

describe("AUDIT fixes", () => {
  test("CSP closes form-action and base-uri (default-src does NOT cover them)", () => {
    const out = withCsp("<p>x</p>");
    expect(out).toContain("form-action 'none'");
    expect(out).toContain("base-uri 'none'");
    expect(out).toContain("default-src 'none'");
  });

  test("same-second artifacts do not overwrite each other", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-artifact5-"));
    try {
      const a = await artifactTool({ title: "Same Name", html: "<p>first</p>" }, ctxFor(dir));
      const b = await artifactTool({ title: "Same Name", html: "<p>second</p>" }, ctxFor(dir));
      expect(a).not.toBe(b);
      const files = readdirSync(join(dir, "artifacts"));
      expect(files.length).toBe(2);
      const bodies = files.map((f) => readFileSync(join(dir, "artifacts", f), "utf8"));
      expect(bodies.some((x) => x.includes("first"))).toBe(true);
      expect(bodies.some((x) => x.includes("second"))).toBe(true);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});
