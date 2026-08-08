// Data-integrity regressions from the core audit: a crash-torn session used to
// eat the next message forever, and compaction orphaned the whole undo trail.
import { describe, expect, test } from "bun:test";
import { appendFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Checkpoints } from "../src/checkpoint.ts";
import { loadConfig } from "../src/config.ts";
import { Session } from "../src/session.ts";

const dataDir = () => mkdtempSync(join(tmpdir(), "ap-persist-"));

describe("torn session recovery", () => {
  test("a message appended after a crash is NOT swallowed by the torn line", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    s.append({ role: "user", content: "first" });
    // Simulate a kill mid-write: a fragment with no trailing newline.
    appendFileSync(join(dir, "sessions", `${s.id}.jsonl`), '{"t":"msg","role":"user","con');

    const resumed = Session.load(dir, s.id);
    expect(resumed.history.length).toBe(1);
    expect(resumed.recovered).toBe(true);
    resumed.append({ role: "user", content: "written after the crash" });

    // The reload is what used to lose it: the fragment + the new record had
    // fused into one unparsable line.
    const again = Session.load(dir, s.id);
    expect(again.history.length).toBe(2);
    expect((again.history[1] as any).content).toBe("written after the crash");
  });

  test("the repair does not corrupt later appends", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    appendFileSync(join(dir, "sessions", `${s.id}.jsonl`), '{"torn');
    const r = Session.load(dir, s.id);
    r.append({ role: "user", content: "a" });
    r.append({ role: "assistant", content: "b" });
    const again = Session.load(dir, s.id);
    expect(again.history.map((m: any) => m.content)).toEqual(["a", "b"]);
    expect(again.recovered).toBe(true); // the damaged fragment is still visible as skipped
  });

  test("loading never rewrites the file (mtime ordering must not change)", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    appendFileSync(join(dir, "sessions", `${s.id}.jsonl`), "{broken");
    const before = readFileSync(join(dir, "sessions", `${s.id}.jsonl`), "utf8");
    Session.load(dir, s.id);
    Session.load(dir, s.id);
    expect(readFileSync(join(dir, "sessions", `${s.id}.jsonl`), "utf8")).toBe(before);
  });

  test("a clean session reports no recovery", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    s.append({ role: "user", content: "x" });
    expect(Session.load(dir, s.id).recovered).toBe(false);
  });
});

describe("session meta", () => {
  test("checkpointId round-trips so /undo survives compaction + restart", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t", checkpointId: "original-session-id" });
    s.append({ role: "user", content: "after compaction" });
    const loaded = Session.load(dir, s.id);
    expect(loaded.meta?.checkpointId).toBe("original-session-id");
  });

  test("sessions without the field are unaffected", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    expect(Session.load(dir, s.id).meta?.checkpointId).toBeUndefined();
  });

  test("rename appends title meta (load last-wins) and delete removes the file", () => {
    const dir = dataDir();
    const s = Session.create(dir, { cwd: ".", model: "m", at: "t" });
    s.append({ role: "user", content: "hi" });
    Session.rename(dir, s.id, "  my feature work  ");
    const loaded = Session.load(dir, s.id);
    expect(loaded.meta?.title).toBe("my feature work");
    expect(loaded.history.length).toBe(1);

    Session.rename(dir, s.id, ""); // clear
    expect(Session.load(dir, s.id).meta?.title).toBeUndefined();

    Session.delete(dir, s.id);
    expect(() => Session.load(dir, s.id)).toThrow();
    expect(Session.list(dir).some((x) => x.id === s.id)).toBe(false);
  });
});

describe("checkpoint work-tree binding", () => {
  test("the work-tree is snapshotted, not read from config.cwd at call time", () => {
    const work = mkdtempSync(join(tmpdir(), "ap-work-"));
    writeFileSync(join(work, "a.txt"), "one\n");
    const config = loadConfig({ cwd: work } as any);
    config.dataDir = dataDir();
    const cp = new Checkpoints(config, "sess-1");
    if (!cp.available()) return; // no git on this machine
    expect(cp.commit("first")).toBeTruthy();

    // /worktree mutates config.cwd in place; an unbound Checkpoints would now
    // commit a DIFFERENT directory into this same history.
    const other = mkdtempSync(join(tmpdir(), "ap-other-"));
    writeFileSync(join(other, "b.txt"), "two\n");
    config.cwd = other;

    writeFileSync(join(work, "a.txt"), "one changed\n");
    const hash = cp.commit("second");
    expect(hash).toBeTruthy();
    const files = cp.filesChanged(null).map((f) => f.file);
    expect(files).toContain("a.txt");
    expect(files).not.toContain("b.txt"); // the foreign tree never entered
  });
});
