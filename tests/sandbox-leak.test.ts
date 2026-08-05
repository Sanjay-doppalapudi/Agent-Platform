// Security regressions found by the core audit. Each of these was exploitable:
// grep/glob printed credentials that `read` denies, Windows device prefixes
// walked straight past every containment check, and the agent tool accepted
// any cwd as a brand-new sandbox root.
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { execTool } from "../src/tools/index.ts";
import { isPrivatePath, privateExcludeGlobs, resolvePath } from "../src/tools/shared.ts";

const SECRET = "sk-or-LEAK-CANARY-4242";

function workspace() {
  const cwd = mkdtempSync(join(tmpdir(), "ap-leak-"));
  const dataDir = join(cwd, ".ap");
  mkdirSync(join(dataDir, "sessions"), { recursive: true });
  mkdirSync(join(dataDir, "checkpoints"), { recursive: true });
  writeFileSync(join(dataDir, "credentials.json"), JSON.stringify({ openrouter: SECRET }));
  writeFileSync(join(dataDir, "sessions", "s1.jsonl"), `{"role":"user","content":"${SECRET} transcript"}\n`);
  writeFileSync(join(cwd, "app.txt"), "hello world\n");
  const config = loadConfig({ cwd } as any);
  config.dataDir = dataDir;
  return { cwd, dataDir, config };
}
const ctxFor = (w: ReturnType<typeof workspace>) =>
  ({ cwd: w.cwd, config: w.config, signal: new AbortController().signal, permit: async () => true }) as any;

describe("grep/glob must not walk into AP-private data", () => {
  test("grep cannot surface credentials even with an allow-all permit", async () => {
    const w = workspace();
    const r = await execTool("grep", JSON.stringify({ pattern: "sk-or-LEAK-CANARY" }), ctxFor(w));
    expect(r.output).not.toContain(SECRET);
    expect(r.output).not.toContain("credentials.json");
  });

  test("grep cannot surface session transcripts", async () => {
    const w = workspace();
    const r = await execTool("grep", JSON.stringify({ pattern: "transcript" }), ctxFor(w));
    expect(r.output).not.toContain("s1.jsonl");
  });

  test("glob cannot enumerate private files", async () => {
    const w = workspace();
    const r = await execTool("glob", JSON.stringify({ pattern: ".ap/**" }), ctxFor(w));
    expect(r.output).not.toContain("credentials.json");
    expect(r.output).not.toContain("s1.jsonl");
  });

  test("ordinary workspace search is unaffected", async () => {
    const w = workspace();
    const g = await execTool("grep", JSON.stringify({ pattern: "hello world" }), ctxFor(w));
    expect(g.output).toContain("app.txt");
    const files = await execTool("glob", JSON.stringify({ pattern: "*.txt" }), ctxFor(w));
    expect(files.output).toContain("app.txt");
  });

  test("exclusion globs use forward slashes and cover file + directory forms", () => {
    const w = workspace();
    const globs = privateExcludeGlobs(w.config, w.cwd).filter((g) => g !== "-g");
    expect(globs.some((g) => g === "!.ap/credentials.json")).toBe(true);
    expect(globs.some((g) => g === "!.ap/sessions/**")).toBe(true);
    for (const g of globs) expect(g).not.toContain("\\"); // backslash = escape in rg
  });

  test("no exclusions are emitted when the data dir is outside the search root", () => {
    const w = workspace();
    const elsewhere = mkdtempSync(join(tmpdir(), "ap-other-"));
    expect(privateExcludeGlobs(w.config, elsewhere)).toEqual([]);
  });
});

describe("Windows device prefixes cannot bypass containment", () => {
  const w = workspace();
  const target = join(w.dataDir, "credentials.json");

  test.if(process.platform === "win32")("\\\\?\\ prefix still resolves as private", () => {
    expect(isPrivatePath(resolvePath(`\\\\?\\${target}`, w.cwd), w.config)).toBe(true);
  });
  test.if(process.platform === "win32")("\\\\.\\ prefix still resolves as private", () => {
    expect(isPrivatePath(resolvePath(`\\\\.\\${target}`, w.cwd), w.config)).toBe(true);
  });
  test("plain path is private (control)", () => {
    expect(isPrivatePath(resolvePath(target, w.cwd), w.config)).toBe(true);
  });
  test("read denies the device-prefixed path", async () => {
    const w2 = workspace();
    const p = process.platform === "win32"
      ? `\\\\?\\${join(w2.dataDir, "credentials.json")}`
      : join(w2.dataDir, "credentials.json");
    const r = await execTool("read", JSON.stringify({ path: p }), ctxFor(w2));
    expect(r.error).toBe(true);
    expect(r.output).not.toContain(SECRET);
  });
});

describe("agent tool cwd is sandbox-gated", () => {
  test("a cwd outside the workspace is denied when the permit says no", async () => {
    const w = workspace();
    const outside = mkdtempSync(join(tmpdir(), "ap-outside-"));
    const denyCtx = { ...ctxFor(w), permit: async () => false };
    const r = await execTool("agent", JSON.stringify({ task: "noop", cwd: outside }), denyCtx);
    expect(r.error).toBe(true);
    expect(r.output.toLowerCase()).toContain("denied");
  });

  test("AP-private cwd is refused even with an allow-all permit", async () => {
    const w = workspace();
    const r = await execTool("agent", JSON.stringify({ task: "noop", cwd: join(w.dataDir, "sessions") }), ctxFor(w));
    expect(r.error).toBe(true);
    expect(r.output).toContain("AP-private");
  });
});
