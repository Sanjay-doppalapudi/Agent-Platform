// Background task delegation: agent {background:true} registers here and the
// subagent runs detached from the turn that started it. Completion queues a
// note that the REPL folds into the NEXT user message (the loop.ts pending
// pattern) — the model finds out results the next time it runs, and nothing
// is ever injected mid-turn. In-process only: tasks die with the process, and
// <dataDir>/tasks.jsonl (append-only, torn-line tolerant, bg.ts-style) keeps
// an audit trail so a later `/tasks` can say what happened.
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Config } from "./config.ts";

export interface TaskInfo {
  id: number;
  task: string;
  status: "running" | "done" | "error" | "killed";
  startedAt: number;
  endedAt?: number;
  steps: number;
  /** Final text (done) or failure clue (error), truncated. */
  result?: string;
  ctrl: AbortController;
  /** Absolute ms deadline (startedAt + the task's own timeout). settleTasks
   *  waits until the LATEST deadline, so a long task is never pre-empted by a
   *  shorter fixed wait. */
  deadline: number;
}

const tasks: TaskInfo[] = [];
const notes: string[] = [];
let nextId = 1;

export function listTasks(): TaskInfo[] {
  return tasks;
}

export function taskById(id: number): TaskInfo | undefined {
  return tasks.find((t) => t.id === id);
}

export function registerTask(task: string, timeoutMs = 300_000): TaskInfo {
  const startedAt = Date.now();
  const info: TaskInfo = {
    id: nextId++,
    task: task.replace(/\s+/g, " ").slice(0, 100),
    status: "running",
    startedAt,
    steps: 0,
    ctrl: new AbortController(),
    deadline: startedAt + timeoutMs,
  };
  tasks.push(info);
  return info;
}

export function finishTask(info: TaskInfo, status: "done" | "error" | "killed", result: string): void {
  info.status = status;
  info.endedAt = Date.now();
  info.result = result.slice(0, 20_000);
  const secs = ((info.endedAt - info.startedAt) / 1000).toFixed(1);
  notes.push(
    `[background task #${info.id} ${status} after ${secs}s · ${info.steps} steps] ${info.task}\n${info.result || "(no output)"}`,
  );
}

/** Completed-task notes queued since the last turn. Draining is destructive —
 *  each note reaches the model exactly once. */
export function drainTaskNotes(): string[] {
  return notes.splice(0, notes.length);
}

/** True while anything is still running — run.ts awaits this before exit
 *  (like lifecycleSettled), because exiting kills children on Windows. */
export function tasksRunning(): boolean {
  return tasks.some((t) => t.status === "running");
}

/**
 * Wait for every running task, bounded by the LATEST task deadline (not a
 * fixed window): a fixed 300s wait pre-empted tasks that legitimately had up
 * to 900s, and process.exit then hard-killed the child, dropping its result.
 * If a task somehow outlives its own deadline, abort it gracefully so the
 * runner's kill path records a note and an audit "end" line — instead of the
 * process dying underneath it and freezing the task at "running" forever.
 */
export async function settleTasks(graceMs = 5_000): Promise<void> {
  while (tasksRunning()) {
    const latest = Math.max(...tasks.filter((t) => t.status === "running").map((t) => t.deadline));
    if (Date.now() > latest + graceMs) break;
    await new Promise((r) => setTimeout(r, 200));
  }
  // Anything still running is past its own deadline: abort, then give the
  // runner a moment to finish its .then() (note + audit end line).
  const stuck = tasks.filter((t) => t.status === "running");
  if (stuck.length) {
    for (const t of stuck) t.ctrl.abort();
    const until = Date.now() + graceMs;
    while (tasksRunning() && Date.now() < until) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
}

/** Append one audit line; never on the startup path, never throws. */
export function auditTask(config: Config, info: TaskInfo, event: "start" | "end"): void {
  try {
    const p = join(config.dataDir, "tasks.jsonl");
    mkdirSync(dirname(p), { recursive: true });
    const line = {
      at: new Date().toISOString(),
      event,
      id: info.id,
      pidNs: process.pid, // ids restart per process; pid disambiguates lines
      task: info.task,
      status: info.status,
      steps: info.steps,
      result: event === "end" ? (info.result ?? "").slice(0, 2000) : undefined,
    };
    appendFileSync(p, JSON.stringify(line) + "\n");
  } catch {}
}
