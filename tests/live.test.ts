// Cross-process live status: the snapshots that let `ap watch` in one
// terminal see the agents/tasks/flows running inside another ap process.
import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { clearLive, formatSnapshot, publishLive, readLive, pidAlive } from "../src/live.ts";

const dirp = () => mkdtempSync(join(tmpdir(), "ap-live-"));
const snap = (over: Partial<Parameters<typeof publishLive>[1]> = {}) => ({
  cwd: "C:/work", session: "s1", model: "openrouter/x", busy: true, agents: [], ...over,
});

describe("publish / read", () => {
  test("a published snapshot is readable and carries this pid", () => {
    const d = dirp();
    try {
      publishLive(d, snap({ ctxPct: 42 }), true);
      const all = readLive(d);
      expect(all).toHaveLength(1);
      expect(all[0]!.pid).toBe(process.pid);
      expect(all[0]!.ctxPct).toBe(42);
      expect(all[0]!.model).toBe("openrouter/x");
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("writes are throttled unless forced (publishing must not cost a turn)", () => {
    const d = dirp();
    try {
      publishLive(d, snap({ session: "first" }), true);
      publishLive(d, snap({ session: "second" })); // throttled away
      expect(readLive(d)[0]!.session).toBe("first");
      publishLive(d, snap({ session: "third" }), true); // forced through
      expect(readLive(d)[0]!.session).toBe("third");
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("clearLive removes this process's snapshot", () => {
    const d = dirp();
    try {
      publishLive(d, snap(), true);
      expect(readLive(d)).toHaveLength(1);
      clearLive(d);
      expect(readLive(d)).toHaveLength(0);
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("no directory → no snapshots, no throw", () => {
    const d = join(tmpdir(), "ap-live-nonexistent-xyz");
    expect(readLive(d)).toEqual([]);
  });
});

describe("self-cleaning", () => {
  test("a snapshot from a DEAD pid is pruned on read (crash cleanup)", () => {
    const d = dirp();
    try {
      mkdirSync(join(d, "live"), { recursive: true });
      // pid 0x7FFFFFFF is not a real process on any platform we target.
      const dead = 0x7ffffffe;
      writeFileSync(join(d, "live", `${dead}.json`), JSON.stringify({ pid: dead, at: Date.now(), ...snap() }));
      expect(readLive(d)).toHaveLength(0);
      expect(readdirSync(join(d, "live"))).toHaveLength(0); // file removed
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("a STALE snapshot from a live pid is pruned too (hung/paused writer)", () => {
    const d = dirp();
    try {
      mkdirSync(join(d, "live"), { recursive: true });
      writeFileSync(join(d, "live", `${process.pid}.json`),
        JSON.stringify({ pid: process.pid, at: Date.now() - 120_000, ...snap() }));
      expect(readLive(d)).toHaveLength(0);
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("a corrupt snapshot is dropped, not thrown", () => {
    const d = dirp();
    try {
      mkdirSync(join(d, "live"), { recursive: true });
      writeFileSync(join(d, "live", "999999.json"), "{not json");
      expect(() => readLive(d)).not.toThrow();
      expect(readLive(d)).toHaveLength(0);
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("no .tmp file survives a publish (atomic rename)", () => {
    const d = dirp();
    try {
      publishLive(d, snap(), true);
      expect(readdirSync(join(d, "live")).some((f) => f.endsWith(".tmp"))).toBe(false);
    } finally { rmSync(d, { recursive: true, force: true }); }
  });

  test("pidAlive is true for us and false for an absurd pid", () => {
    expect(pidAlive(process.pid)).toBe(true);
    expect(pidAlive(0x7ffffffe)).toBe(false);
  });
});

describe("formatSnapshot", () => {
  const now = 1_000_000;
  test("summarises model, state, context and cost", () => {
    const lines = formatSnapshot({ pid: 7, at: now, cwd: "C:/w", session: "s", model: "p/m", busy: true, ctxPct: 12, usd: 0.0312, agents: [] }, now);
    expect(lines[0]).toContain("pid 7");
    expect(lines[0]).toContain("working");
    expect(lines[0]).toContain("ctx 12%");
    expect(lines[0]).toContain("~$0.031");
    expect(lines[1]).toContain("C:/w");
  });
  test("idle processes say so", () => {
    expect(formatSnapshot({ pid: 7, at: now, cwd: "c", session: "s", model: "m", busy: false, agents: [] }, now)[0]).toContain("idle");
  });
  test("a running flow and its agents each get a line", () => {
    const lines = formatSnapshot({
      pid: 7, at: now, cwd: "c", session: "s", model: "m", busy: true, flow: "review", flowStep: "agent 2/5",
      agents: [{ id: 1, label: "audit foo.ts", status: "running", steps: 3, startedAt: now - 5000, background: true }],
    }, now);
    expect(lines.some((l) => l.includes("◆ flow review") && l.includes("agent 2/5"))).toBe(true);
    const agentLine = lines.find((l) => l.includes("#1"))!;
    expect(agentLine).toContain("[running]");
    expect(agentLine).toContain("&"); // background marker
    expect(agentLine).toContain("audit foo.ts");
    expect(agentLine).toContain("5s");
  });
});
