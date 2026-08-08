// Discovery-layer regressions. Both were security-relevant: a crafted skill
// repo could write over the user's own config/credentials, and a BOM silently
// voided an agent's tool restrictions.
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { clearAgentCache, discoverAgents, getAgent } from "../src/agents.ts";
import { loadConfig } from "../src/config.ts";
import { clearSkillCache, discoverSkills, safeSkillPath } from "../src/skills.ts";

describe("skills installer path validation", () => {
  // A GitHub tree path is always '/'-separated, so a backslash is data — but
  // on Windows join() treats it as a separator, landing "..\..\config.json"
  // on <dataDir>/config.json, whose hooks.onDone is a shell command.
  const unsafe = [
    "..\\..\\config.json",
    "..\\credentials.json",
    "a/../../escape.md",
    "../outside.md",
    "/etc/passwd",
    "C:\\Windows\\Temp\\pwn.txt",
    "sub/../../../x",
    "",
    "a/./b",
    "bad\0name",
  ];
  for (const rel of unsafe) {
    test(`rejects ${JSON.stringify(rel)}`, () => expect(safeSkillPath(rel)).toBe(false));
  }

  const safe = ["SKILL.md", "sub/dir/file.md", "a-b_c.1.md", "deep/nested/path/x.txt"];
  for (const rel of safe) {
    test(`allows ${JSON.stringify(rel)}`, () => expect(safeSkillPath(rel)).toBe(true));
  }
});

describe("BOM must not void frontmatter", () => {
  function withAgent(content: string) {
    const cwd = mkdtempSync(join(tmpdir(), "ap-bomfm-"));
    mkdirSync(join(cwd, ".ap", "agents"), { recursive: true });
    writeFileSync(join(cwd, ".ap", "agents", "reviewer.md"), content);
    return loadConfig({ cwd } as any);
  }
  const body = "---\ndescription: read-only reviewer\nmodel: acme/fast\ntools: read, grep\n---\nYou never write files.\n";

  test("without a BOM the restrictions apply (control)", () => {
    const a = getAgent(withAgent(body), "reviewer")!;
    expect(a.tools).toEqual(["read", "grep"]);
  });

  // REGRESSION: a BOM made the `---` opener fail to match, so tools: was
  // dropped and a "read-only reviewer" ran with write/edit/bash.
  test("with a BOM the restrictions STILL apply", () => {
    const a = getAgent(withAgent("\uFEFF" + body), "reviewer")!;
    expect(a.tools).toEqual(["read", "grep"]);
    expect(a.model).toBe("acme/fast");
    expect(a.description).toBe("read-only reviewer");
    expect(a.body).not.toContain("---"); // frontmatter consumed, not injected as text
  });

  test("skills with a BOM are still described", () => {
    const cwd = mkdtempSync(join(tmpdir(), "ap-bomskill-"));
    mkdirSync(join(cwd, ".ap", "skills", "demo"), { recursive: true });
    writeFileSync(
      join(cwd, ".ap", "skills", "demo", "SKILL.md"),
      "\uFEFF---\nname: demo\ndescription: a demo skill\n---\nBody here.\n",
    );
    const s = discoverSkills(loadConfig({ cwd } as any))[0]!;
    expect(s.name).toBe("demo");
    expect(s.description).toBe("a demo skill");
  });

  test("nested skill folders are discovered", () => {
    clearSkillCache();
    const cwd = mkdtempSync(join(tmpdir(), "ap-nestskill-"));
    mkdirSync(join(cwd, ".ap", "skills", "group", "nested"), { recursive: true });
    writeFileSync(
      join(cwd, ".ap", "skills", "group", "nested", "SKILL.md"),
      "---\nname: nested\ndescription: deep skill\n---\nBody.\n",
    );
    const s = discoverSkills(loadConfig({ cwd } as any)).find((x) => x.name === "nested");
    expect(s?.description).toBe("deep skill");
  });

  test("a BOM'd skill marked internal is still skipped", () => {
    const cwd = mkdtempSync(join(tmpdir(), "ap-bomint-"));
    mkdirSync(join(cwd, ".ap", "skills", "hidden"), { recursive: true });
    writeFileSync(
      join(cwd, ".ap", "skills", "hidden", "SKILL.md"),
      "\uFEFF---\nname: hidden\ninternal: true\n---\nBody.\n",
    );
    expect(discoverSkills(loadConfig({ cwd } as any)).map((s) => s.name)).not.toContain("hidden");
  });
});

describe("discovery cache keys", () => {
  // POSIX only: this pins that a `|`-containing path cannot forge a cache-key
  // collision, which needs REAL directories whose names contain "|". Windows
  // forbids "|" in filenames outright (mkdir fails with ENOENT), so the
  // collision is impossible there by construction and the test would only be
  // asserting that mkdir failed.
  test.skipIf(process.platform === "win32")("paths containing the former delimiter cannot collide", () => {
    const root = mkdtempSync(join(tmpdir(), "ap-cache-key-"));
    const firstData = `${root}/data|${root}/middle`;
    const firstCwd = `${root}/tail`;
    const secondData = `${root}/data`;
    const secondCwd = `${root}/middle|${root}/tail`;
    // `${dataDir}|${cwd}` collides for this pair; JSON tuple encoding does not.
    mkdirSync(join(firstData, "agents"), { recursive: true });
    mkdirSync(join(firstData, "skills", "first"), { recursive: true });
    writeFileSync(join(firstData, "agents", "first.md"), "First agent\n");
    writeFileSync(join(firstData, "skills", "first", "SKILL.md"), "---\nname: first\n---\n");
    clearAgentCache();
    clearSkillCache();

    expect(discoverAgents({ dataDir: firstData, cwd: firstCwd } as any).map((a) => a.name)).toEqual(["first"]);
    expect(discoverAgents({ dataDir: secondData, cwd: secondCwd } as any)).toEqual([]);
    expect(discoverSkills({ dataDir: firstData, cwd: firstCwd } as any).map((s) => s.name)).toEqual(["first"]);
    expect(discoverSkills({ dataDir: secondData, cwd: secondCwd } as any)).toEqual([]);
  });
});

describe("system prompt describes the real platform", () => {
  test("no hardcoded Windows 11 claim", async () => {
    const { buildSystemPrompt } = await import("../src/prompt.ts");
    const cwd = mkdtempSync(join(tmpdir(), "ap-prompt-"));
    const p = buildSystemPrompt(loadConfig({ cwd } as any));
    expect(p).not.toContain("Windows 11");
    const expected = process.platform === "win32" ? "Windows" : process.platform === "darwin" ? "macOS" : "Linux";
    expect(p).toContain(`Environment: ${expected}.`);
  });
});
