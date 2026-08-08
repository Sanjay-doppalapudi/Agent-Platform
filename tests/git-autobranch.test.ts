// Real-git hygiene: autoBranch latch + PR material helpers (no network).
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { Config } from "../src/config.ts";
import {
  defaultPrBase,
  maybeAutoBranch,
  prPromptMaterial,
  resetAutoBranchLatch,
  slugifyBranch,
} from "../src/git.ts";
import { tmuxAvailable, tmuxMissingHint, tmuxSessionName } from "../src/tmux.ts";

function git(cwd: string, args: string[]) {
  return Bun.spawnSync(["git", ...args], { cwd, stdout: "pipe", stderr: "pipe" });
}

describe("tmux helpers", () => {
  test("session names are safe slugs", () => {
    expect(tmuxSessionName("Fix the Login!!")).toBe("ap-fix-the-login");
    expect(tmuxSessionName("")).toMatch(/^ap-/);
  });
  test("missing hint is platform-aware", () => {
    const h = tmuxMissingHint();
    if (process.platform === "win32") expect(h).toContain("Windows");
    else expect(h.length).toBeGreaterThan(10);
  });
  test("tmuxAvailable matches Bun.which", () => {
    expect(tmuxAvailable()).toBe(!!Bun.which("tmux"));
  });
});

describe("maybeAutoBranch", () => {
  const root = join(tmpdir(), `ap-autobranch-${Date.now()}`);
  let cfg: Config;

  beforeAll(() => {
    mkdirSync(root, { recursive: true });
    git(root, ["init", "-b", "main"]);
    git(root, ["config", "user.email", "test@example.com"]);
    git(root, ["config", "user.name", "test"]);
    writeFileSync(join(root, "a.txt"), "1\n");
    git(root, ["add", "."]);
    git(root, ["commit", "-m", "init"]);
    cfg = {
      cwd: root,
      dataDir: join(root, ".ap-data"),
      provider: "",
      providers: {},
      mode: "code",
      permissions: "yolo",
      sandbox: "off",
      bashGuard: "off",
      light: false,
      checkpoints: "off",
      autoCompact: "off",
      autoMemory: "off",
      streamIdleSeconds: 30,
      maxIterations: 5,
      contextBudgetChars: 10_000,
      redactEnv: false,
      shell: "auto",
      parallelPolicy: "safe",
      ignore: [],
      git: { autoBranch: true },
    } as Config;
  });

  afterAll(() => {
    try { rmSync(root, { recursive: true, force: true }); } catch { /* */ }
  });

  test("creates ap/<slug> on protected branch, then latches", () => {
    resetAutoBranchLatch();
    const notes: string[] = [];
    const name = maybeAutoBranch(cfg, "add feature flag", (m) => notes.push(m));
    expect(name).toBe(slugifyBranch("add feature flag"));
    expect(notes.some((n) => n.includes("auto-branch"))).toBe(true);
    const br = git(root, ["branch", "--show-current"]).stdout.toString().trim();
    expect(br).toBe(name);

    // Second call is a no-op (latch).
    const again = maybeAutoBranch(cfg, "other", (m) => notes.push(m));
    expect(again).toBeNull();
  });

  test("light profile never auto-branches", () => {
    resetAutoBranchLatch();
    const light = { ...cfg, light: true, git: { autoBranch: true } };
    expect(maybeAutoBranch(light, "x")).toBeNull();
  });

  test("autoBranch off skips", () => {
    resetAutoBranchLatch();
    expect(maybeAutoBranch({ ...cfg, git: { autoBranch: false } }, "x")).toBeNull();
  });
});

describe("pr helpers", () => {
  test("defaultPrBase returns a non-empty branch name in this repo", () => {
    const cfg = {
      cwd: process.cwd(),
      dataDir: join(tmpdir(), "ap-x"),
      light: false,
    } as Config;
    const b = defaultPrBase(cfg);
    expect(b.length).toBeGreaterThan(0);
  });

  test("prPromptMaterial includes head", () => {
    const cfg = { cwd: process.cwd(), dataDir: join(tmpdir(), "ap-x"), light: false } as Config;
    const m = prPromptMaterial(cfg);
    expect(m.head.length).toBeGreaterThan(0);
    expect(m.base.length).toBeGreaterThan(0);
  });
});
