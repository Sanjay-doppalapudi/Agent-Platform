// Background task delegation: the registry/note plumbing that carries a
// detached subagent's result into the next turn. Pure/local — no child
// processes; runSubagent itself is exercised by the existing agent-tool
// integration paths.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { auditTask, drainTaskNotes, finishTask, listTasks, registerTask, settleTasks, taskById, tasksRunning } from "../src/tasks.ts";

describe("task registry", () => {
  test("register → finish → note reaches the next turn exactly once", () => {
    const t = registerTask("summarize the changelog");
    expect(t.status).toBe("running");
    expect(tasksRunning()).toBe(true);
    expect(taskById(t.id)).toBe(t);

    finishTask(t, "done", "the changelog says: nothing");
    expect(t.status).toBe("done");

    const notes = drainTaskNotes();
    expect(notes.length).toBe(1);
    expect(notes[0]).toContain(`#${t.id} done`);
    expect(notes[0]).toContain("the changelog says: nothing");
    expect(drainTaskNotes()).toEqual([]); // destructive drain — never twice
  });

  test("task text is normalized and capped", () => {
    const t = registerTask("a\n\n  b   c" + "x".repeat(200));
    expect(t.task.startsWith("a b c")).toBe(true);
    expect(t.task.length).toBeLessThanOrEqual(100);
    finishTask(t, "error", "boom");
    drainTaskNotes();
  });

  test("kill uses the task's OWN controller (a turn abort must not reach it)", () => {
    const t = registerTask("long thing");
    expect(t.ctrl.signal.aborted).toBe(false);
    t.ctrl.abort();
    expect(t.ctrl.signal.aborted).toBe(true);
    finishTask(t, "killed", "killed");
    drainTaskNotes();
  });

  test("audit lines are appended JSONL, one per event", () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-tasks-"));
    try {
      const t = registerTask("audit me");
      auditTask({ dataDir: dir } as any, t, "start");
      finishTask(t, "done", "result text");
      auditTask({ dataDir: dir } as any, t, "end");
      const lines = readFileSync(join(dir, "tasks.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
      expect(lines.length).toBe(2);
      expect(lines[0].event).toBe("start");
      expect(lines[1].event).toBe("end");
      expect(lines[1].result).toContain("result text");
      drainTaskNotes();
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("listTasks exposes every registered task", () => {
    const before = listTasks().length;
    const t = registerTask("x");
    expect(listTasks().length).toBe(before + 1);
    finishTask(t, "done", "");
    drainTaskNotes();
  });
});

describe("settleTasks deadline handling (AUDIT: a fixed 300s wait pre-empted 900s tasks)", () => {
  test("registerTask records a deadline derived from the task's OWN timeout", () => {
    const t = registerTask("long one", 900_000);
    expect(t.deadline - t.startedAt).toBe(900_000);
    const d = registerTask("default one");
    expect(d.deadline - d.startedAt).toBe(300_000);
    finishTask(t, "done", ""); finishTask(d, "done", ""); drainTaskNotes();
  });

  test("settleTasks waits for a task whose deadline is still in the future", async () => {
    const t = registerTask("still working", 60_000);
    let settled = false;
    const p = settleTasks(50).then(() => { settled = true; });
    await new Promise((r) => setTimeout(r, 120));
    expect(settled).toBe(false); // deadline far away → still waiting
    finishTask(t, "done", "finished");
    await p;
    expect(settled).toBe(true);
    drainTaskNotes();
  });

  test("a task past its deadline is ABORTED (graceful path) rather than left for process.exit to kill", async () => {
    const t = registerTask("overdue", 1); // deadline ~immediately
    await new Promise((r) => setTimeout(r, 5));
    await settleTasks(30);
    expect(t.ctrl.signal.aborted).toBe(true);
    finishTask(t, "killed", "killed"); drainTaskNotes();
  });

  test("settleTasks returns immediately when nothing is running", async () => {
    const t0 = Date.now();
    await settleTasks(50);
    expect(Date.now() - t0).toBeLessThan(200);
  });
});
