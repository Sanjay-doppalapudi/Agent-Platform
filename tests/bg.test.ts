// Background-process registry: liveness, pruning, tailing, kill guardrails.
import { describe, expect, test } from "bun:test";
import { appendFileSync, existsSync, mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { formatBytes, isAlive, killBackground, listBackground, recordBackground, tailLog } from "../src/bg.ts";

const dataDir = mkdtempSync(join(tmpdir(), "ap-test-bg-"));
const config: any = { dataDir, cwd: dataDir, light: false };

describe("isAlive", () => {
  test("our own pid is alive", () => expect(isAlive(process.pid)).toBe(true));
  test("an implausible pid is not", () => expect(isAlive(0x7ffffff0)).toBe(false));
});

describe("registry", () => {
  test("records and lists newest-first with liveness + log size", () => {
    const log = join(dataDir, "live.log");
    writeFileSync(log, "line one\nline two\n");
    recordBackground(config, { pid: process.pid, cmd: "bun run dev", cwd: dataDir, log });
    recordBackground(config, { pid: 0x7ffffff0, cmd: "old watcher", cwd: dataDir, log });
    const rows = listBackground(config);
    expect(rows.length).toBe(2);
    expect(rows[0]!.cmd).toBe("old watcher"); // newest first
    const live = rows.find((r) => r.pid === process.pid)!;
    expect(live.alive).toBe(true);
    expect(live.bytes).toBeGreaterThan(0);
    expect(rows.find((r) => r.pid === 0x7ffffff0)!.alive).toBe(false);
  });

  test("dead entries older than the retention window are pruned", () => {
    const old = { pid: 0x7ffffff1, cmd: "ancient", cwd: dataDir, log: join(dataDir, "gone.log"), at: new Date(Date.now() - 30 * 24 * 3600 * 1000).toISOString() };
    appendFileSync(join(dataDir, "background.jsonl"), JSON.stringify(old) + "\n");
    expect(listBackground(config).some((r) => r.pid === 0x7ffffff1)).toBe(false);
  });

  test("torn final line is skipped (crash-safety)", () => {
    appendFileSync(join(dataDir, "background.jsonl"), '{"pid":123,"cmd":"tor');
    expect(() => listBackground(config)).not.toThrow();
    expect(listBackground(config).some((r) => r.pid === 123)).toBe(false);
  });
});

describe("tailLog", () => {
  test("returns the last N lines", () => {
    const p = join(dataDir, "many.log");
    writeFileSync(p, Array.from({ length: 100 }, (_, i) => `line ${i}`).join("\n"));
    const out = tailLog(p, 5);
    expect(out.split("\n").length).toBe(5);
    expect(out).toContain("line 99");
    expect(out).not.toContain("line 90");
  });
  test("empty log reads as no output", () => {
    const p = join(dataDir, "empty.log");
    writeFileSync(p, "");
    expect(tailLog(p)).toBe("(no output yet)");
  });
  test("missing log degrades to a message, never throws", () => {
    expect(tailLog(join(dataDir, "nope.log"))).toContain("unavailable");
  });
});

describe("background output capture (regression)", () => {
  // Detached children silently dropped inherited fds on Windows, so every
  // background log was 0 bytes. The shell must open the file itself.
  test("bash background:true captures stdout AND stderr to the log", async () => {
    const { bashTool } = await import("../src/tools/bash.ts");
    const cwd = mkdtempSync(join(tmpdir(), "ap-test-bgrun-"));
    const cfg: any = { dataDir, cwd, light: false, shell: "auto", sandbox: "off", bashGuard: "off" };
    const out = await bashTool(
      { cmd: "echo out-line; echo err-line 1>&2", background: true },
      { cwd, config: cfg, signal: new AbortController().signal, permit: async () => true } as any,
    );
    const log = out.match(/log=(.+)$/)?.[1]!;
    expect(log).toBeTruthy();
    for (let i = 0; i < 40 && (!existsSync(log) || statSync(log).size === 0); i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    const body = readFileSync(log, "utf8");
    expect(body).toContain("out-line");
    expect(body).toContain("err-line");
  }, 15_000);
});

describe("formatBytes", () => {
  // Regression: an empty log must never display as "1KB" — buffered/never-
  // flushed output is a real signal the user needs to see accurately.
  test("zero reads as empty, not 1KB", () => expect(formatBytes(0)).toBe("empty"));
  test("sub-KB shows bytes", () => expect(formatBytes(240)).toBe("240B"));
  test("KB and MB scale", () => {
    expect(formatBytes(2048)).toBe("2KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0MB");
  });
  test("missing log", () => expect(formatBytes(null)).toBe("no log"));
});

describe("killBackground", () => {
  test("refuses pids AP did not start", () => {
    const r = killBackground(config, 999999);
    expect(r.ok).toBe(false);
    expect(r.message).toContain("not an AP background process");
  });
  test("reports already-stopped recorded processes", () => {
    const r = killBackground(config, 0x7ffffff0);
    expect(r.ok).toBe(false);
    expect(r.message).toContain("already stopped");
  });
});
