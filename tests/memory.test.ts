import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readMemories, repoMemoryDir, repoMemoryKey } from "../src/memory.ts";

describe("repoMemoryKey", () => {
  test("same git common-dir → same key for main and worktree-like cwd", () => {
    const root = mkdtempSync(join(tmpdir(), "ap-memgit-"));
    Bun.spawnSync(["git", "init"], { cwd: root, stdout: "ignore", stderr: "ignore" });
    Bun.spawnSync(["git", "-C", root, "commit", "--allow-empty", "-m", "init"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    const wt = join(root, "wt-a");
    Bun.spawnSync(["git", "-C", root, "worktree", "add", wt, "-b", "ap-test-wt"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    const k1 = repoMemoryKey(root);
    const k2 = repoMemoryKey(wt);
    expect(k1).toMatch(/^git-[a-f0-9]{16}$/);
    expect(k2).toBe(k1);
    expect(repoMemoryDir(join(root, ".ap"), root)).toContain(k1);
  });

  test("non-git cwd uses cwd- prefix and is path-stable", () => {
    const cwd = mkdtempSync(join(tmpdir(), "ap-memnogit-"));
    const k = repoMemoryKey(cwd);
    expect(k).toMatch(/^cwd-[a-f0-9]{16}$/);
    expect(repoMemoryKey(cwd)).toBe(k);
  });
});

describe("readMemories legacy fallback", () => {
  test("reads keyed dir first", () => {
    const base = mkdtempSync(join(tmpdir(), "ap-memread-"));
    const keyed = join(base, "keyed");
    const legacy = join(base, "legacy");
    mkdirSync(keyed, { recursive: true });
    mkdirSync(legacy, { recursive: true });
    writeFileSync(join(keyed, "a.md"), "Title: keyed\nUser wanted: x\nWhy (guess): y\n");
    writeFileSync(join(legacy, "b.md"), "Title: legacy\nUser wanted: x\nWhy (guess): y\n");
    const out = readMemories(keyed, legacy);
    expect(out).toContain("keyed");
    expect(out).not.toContain("legacy");
  });

  test("falls back to top-level legacy when keyed empty", () => {
    const base = mkdtempSync(join(tmpdir(), "ap-memfb-"));
    const keyed = join(base, "keyed");
    const legacy = join(base, "legacy");
    mkdirSync(keyed, { recursive: true });
    mkdirSync(legacy, { recursive: true });
    writeFileSync(join(legacy, "old.md"), "Title: old\nUser wanted: x\nWhy (guess): y\n");
    const out = readMemories(keyed, legacy);
    expect(out).toContain("old");
  });
});
