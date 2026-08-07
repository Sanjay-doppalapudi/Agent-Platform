// The interactive viewer: frame layout at both levels (process list and
// subagent detail) plus the bottom-right legend. renderFrame is pure, so all
// of it is assertable without a terminal.
import { describe, expect, test } from "bun:test";
import { renderFrame, type ViewState } from "../src/watch.ts";
import { setTheme } from "../src/theme.ts";
import type { LiveSnapshot } from "../src/live.ts";

setTheme("mono"); // no SGR noise in assertions

const NOW = 2_000_000;
const view = (over: Partial<ViewState> = {}): ViewState => ({ proc: 0, agent: 0, detail: false, scroll: 0, ...over });
const agent = (over = {}) => ({
  id: 1, label: "Say hi from subagent 1", status: "done", steps: 0, startedAt: NOW - 37_000, ...over,
});
const snap = (over: Partial<LiveSnapshot> = {}): LiveSnapshot => ({
  pid: 100, at: NOW, cwd: "C:/work", session: "sess1", model: "prov/model",
  busy: true, ctxPct: 5, agents: [], ...over,
});
const plain = (lines: string[]) => lines.map((l) => l.replace(/\x1b\[[0-9;]*m/g, ""));

describe("frame geometry", () => {
  test("always fills exactly the terminal height", () => {
    for (const rows of [10, 24, 40]) {
      expect(renderFrame([snap()], view(), rows, 80, NOW)).toHaveLength(rows);
    }
  });
  test("no line exceeds the terminal width", () => {
    const wide = snap({ cwd: "C:/" + "x".repeat(200), agents: [agent({ label: "y".repeat(300) })] });
    for (const l of plain(renderFrame([wide], view(), 24, 40, NOW))) {
      expect(l.length).toBeLessThanOrEqual(40);
    }
  });
  test("the legend sits on the LAST row, flush right", () => {
    const f = plain(renderFrame([snap()], view(), 20, 70, NOW));
    const last = f[f.length - 1]!;
    expect(last.endsWith("Esc back")).toBe(true);
    expect(last.startsWith(" ")).toBe(true);
  });
});

describe("process list", () => {
  test("one tab per process, selected one inverted", () => {
    const raw = renderFrame([snap({ pid: 1 }), snap({ pid: 2 })], view({ proc: 1 }), 24, 80, NOW);
    expect(plain([raw[1]!])[0]).toContain("pid 1");
    expect(raw[1]).toContain("\x1b[7m");
  });
  test("the current process is named, not shown as a bare pid", () => {
    expect(plain(renderFrame([snap({ pid: 4242 })], view(), 24, 80, NOW, 4242))[1]).toContain("this session");
  });
  test("empty state explains itself and offers a usable legend", () => {
    const f = plain(renderFrame([], view(), 20, 70, NOW)).join("\n");
    expect(f).toContain("No ap process is publishing status");
    expect(f).toContain("Esc back");
  });
  test("a process with nothing running explains what WOULD appear here", () => {
    const f = plain(renderFrame([snap()], view(), 24, 80, NOW)).join("\n");
    expect(f).toContain("nothing running here yet");
    expect(f).toContain("bash background:true"); // names the other mechanism
  });
});

describe("subagent list (the reported 'cannot view them completely')", () => {
  const s = snap({
    agents: [
      agent({ id: 1, result: "Hi! I am subagent one.\nsecond line" }),
      agent({ id: 2, label: "Say hi from subagent 2", result: "Hi from two." }),
    ],
  });
  test("every agent is listed, the selected one highlighted", () => {
    const raw = renderFrame([s], view({ agent: 1 }), 24, 90, NOW);
    const f = plain(raw).join("\n");
    expect(f).toContain("#1 [done]");
    expect(f).toContain("#2 [done]");
    expect(raw.filter((l) => l.includes("\x1b[7m")).length).toBeGreaterThan(0);
  });
  test("the selected agent's first output line previews inline", () => {
    const f = plain(renderFrame([s], view({ agent: 0 }), 24, 90, NOW)).join("\n");
    expect(f).toContain("Hi! I am subagent one.");
    expect(f).toContain("Enter for the full output");
  });
  test("the legend advertises subagent navigation", () => {
    const f = plain(renderFrame([s], view(), 24, 90, NOW));
    expect(f[f.length - 1]).toContain("↑/↓ subagent");
    expect(f[f.length - 1]).toContain("Enter detail");
  });
});

describe("detail pane", () => {
  const s = snap({
    agents: [agent({ id: 7, fullTask: "Say hi and explain yourself", result: "line1\nline2\nline3" })],
  });
  test("shows the agent's task AND full output", () => {
    const f = plain(renderFrame([s], view({ detail: true }), 24, 80, NOW)).join("\n");
    expect(f).toContain("subagent #7");
    expect(f).toContain("Say hi and explain yourself");
    expect(f).toContain("line1");
    expect(f).toContain("line3");
  });
  test("a still-running agent with nothing streamed yet says so", () => {
    const running = snap({ agents: [agent({ status: "running", result: undefined })] });
    expect(plain(renderFrame([running], view({ detail: true }), 24, 80, NOW)).join("\n"))
      .toContain("nothing streamed yet");
  });
  test("long output scrolls, and the legend reports the position", () => {
    const many = snap({ agents: [agent({ result: Array.from({ length: 100 }, (_, i) => `row${i}`).join("\n") })] });
    const top = plain(renderFrame([many], view({ detail: true, scroll: 0 }), 20, 80, NOW));
    expect(top.join("\n")).toContain("row0");
    expect(top[top.length - 1]).toMatch(/↑\/↓ scroll 1-\d+\/\d+/);
    const down = plain(renderFrame([many], view({ detail: true, scroll: 40 }), 20, 80, NOW));
    expect(down.join("\n")).toContain("row40");
    expect(down.join("\n")).not.toContain("row0\n");
  });
  test("scrolling past the end clamps to the last page", () => {
    const many = snap({ agents: [agent({ result: Array.from({ length: 30 }, (_, i) => `row${i}`).join("\n") })] });
    const f = plain(renderFrame([many], view({ detail: true, scroll: 9999 }), 20, 80, NOW));
    expect(f).toHaveLength(20);
    expect(f.join("\n")).toContain("row29");
  });
  test("long output lines are wrapped, not truncated away", () => {
    const long = snap({ agents: [agent({ result: "z".repeat(300) })] });
    const f = plain(renderFrame([long], view({ detail: true }), 24, 60, NOW));
    expect(f.filter((l) => l.includes("z")).length).toBeGreaterThan(1);
  });
});

describe("robustness", () => {
  test("selection past the end clamps instead of rendering undefined", () => {
    const f = plain(renderFrame([snap({ pid: 1, agents: [agent()] })], view({ proc: 9, agent: 9 }), 24, 80, NOW)).join("\n");
    expect(f).not.toContain("undefined");
  });
  test("detail with nothing selected falls back to the list rather than crashing", () => {
    const f = plain(renderFrame([snap()], view({ detail: true }), 24, 80, NOW)).join("\n");
    expect(f).not.toContain("undefined");
    expect(f).toContain("nothing running here yet");
  });
  test("header pluralisation", () => {
    expect(plain(renderFrame([snap()], view(), 24, 80, NOW))[0]).toContain("1 process ");
    expect(plain(renderFrame([snap({ pid: 1 }), snap({ pid: 2 })], view(), 24, 80, NOW))[0]).toContain("2 processes");
  });
});

describe("background shell processes (bash background:true)", () => {
  // The reported case: the model started a background PROCESS, not a
  // subagent, and the viewer had nothing to show.
  const withProc = snap({
    procs: [{ pid: 19200, cmd: "echo 'thinking...'; sleep 60", log: "C:/logs/bg-1.log", alive: true, bytes: 12 }],
  });

  test("a background process is listed even with no subagents", () => {
    const f = plain(renderFrame([withProc], view(), 24, 90, NOW)).join("\n");
    expect(f).toContain("pid 19200");
    expect(f).toContain("[running]");
    expect(f).toContain("sleep 60");
    expect(f).not.toContain("nothing running here yet");
  });

  test("exited processes and log size are shown", () => {
    const dead = snap({ procs: [{ pid: 5, cmd: "x", log: "l", alive: false, bytes: 4096 }] });
    const f = plain(renderFrame([dead], view(), 24, 90, NOW)).join("\n");
    expect(f).toContain("[exited]");
    expect(f).toContain("4KB");
  });

  test("selection spans agents THEN processes (one flat list)", () => {
    const both = snap({
      agents: [agent({ id: 1 })],
      procs: [{ pid: 42, cmd: "sleep 60", log: "l", alive: true, bytes: 0 }],
    });
    const onProc = renderFrame([both], view({ agent: 1 }), 24, 90, NOW);
    const inverted = onProc.find((l) => l.includes("\x1b[7m") && l.includes("pid 42"));
    expect(inverted).toBeDefined(); // index 1 selects the PROCESS, not the agent
  });

  test("process detail shows the command and reads the log from disk", () => {
    const f = plain(renderFrame([withProc], view({ agent: 0, detail: true }), 24, 90, NOW)).join("\n");
    expect(f).toContain("process 19200");
    expect(f).toContain("command");
    expect(f).toContain("sleep 60");
    expect(f).toContain("log · ");
    expect(f).toContain("log unavailable"); // path does not exist in this test
    expect(f).toContain("pid 100"); // SAME process-tab chrome as the list view
  });
});

describe("readLogTail", () => {
  test("reads the tail and reports a missing file instead of throwing", async () => {
    const { mkdtempSync, writeFileSync, rmSync } = await import("node:fs");
    const { join } = await import("node:path");
    const { tmpdir } = await import("node:os");
    const { readLogTail } = await import("../src/watch.ts");
    const d = mkdtempSync(join(tmpdir(), "ap-log-"));
    try {
      const p = join(d, "x.log");
      writeFileSync(p, Array.from({ length: 500 }, (_, i) => `line${i}`).join("\n"));
      const tail = readLogTail(p, 10);
      expect(tail.split("\n")).toHaveLength(10);
      expect(tail).toContain("line499");
      expect(tail).not.toContain("line100");
      writeFileSync(join(d, "empty.log"), "");
      expect(readLogTail(join(d, "empty.log"))).toContain("empty");
      expect(readLogTail(join(d, "nope.log"))).toContain("unavailable");
    } finally { rmSync(d, { recursive: true, force: true }); }
  });
});

describe("live subagent thinking (reported: 'in the sub process the thinking is not being shown')", () => {
  test("a RUNNING agent shows its streamed reasoning, not a placeholder", () => {
    const s = snap({
      agents: [agent({ status: "running", result: undefined, live: "Let me check the files first...\nreading src/" })],
    });
    const f = plain(renderFrame([s], view({ detail: true }), 24, 80, NOW)).join("\n");
    expect(f).toContain("thinking (live)");
    expect(f).toContain("Let me check the files first");
    expect(f).not.toContain("nothing streamed yet");
  });

  test("a finished agent shows its final output, not the live tail", () => {
    const s = snap({ agents: [agent({ status: "done", live: "partial…", result: "FINAL ANSWER" })] });
    const f = plain(renderFrame([s], view({ detail: true }), 24, 80, NOW)).join("\n");
    expect(f).toContain("output");
    expect(f).toContain("FINAL ANSWER");
    expect(f).not.toContain("thinking (live)");
  });

  test("a just-started agent with nothing streamed says so", () => {
    const s = snap({ agents: [agent({ status: "running", result: undefined, live: undefined })] });
    expect(plain(renderFrame([s], view({ detail: true }), 24, 80, NOW)).join("\n"))
      .toContain("nothing streamed yet");
  });
});
