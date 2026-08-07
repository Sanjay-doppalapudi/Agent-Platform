// REGRESSION (reported): a model emitted malformed tool-call arguments
//   {"path": "research_workflow.py", " "content":…
// The tool failed cleanly, but the RAW string was persisted and re-sent on
// every later request, so the provider rejected the whole conversation with
//   HTTP 400 … "tool arguments must be a stringified JSON object"
// permanently — the session could never take another turn, and the poison
// survived /resume because it was on disk.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Session, sanitizeToolCallArgs } from "../src/session.ts";

const POISON = '{"path": "research_workflow.py", " "content":"x"}'; // the reported shape

function tmpDir() {
  return mkdtempSync(join(tmpdir(), "ap-sess-tc-"));
}
const call = (args: unknown) => ({
  role: "assistant" as const,
  content: null,
  tool_calls: [{ id: "call_1", type: "function", function: { name: "write", arguments: args } }],
});

describe("sanitizeToolCallArgs", () => {
  test("malformed arguments become a valid JSON object (the reported poison)", () => {
    const msg = call(POISON) as any;
    expect(sanitizeToolCallArgs(msg)).toBe(true);
    const out = msg.tool_calls[0].function.arguments;
    expect(() => JSON.parse(out)).not.toThrow();
    expect(typeof JSON.parse(out)).toBe("object");
  });

  test("well-formed arguments are left BYTE-identical (no gratuitous rewrites)", () => {
    const original = '{"path":"a.ts","content":"x"}';
    const msg = call(original) as any;
    expect(sanitizeToolCallArgs(msg)).toBe(false);
    expect(msg.tool_calls[0].function.arguments).toBe(original);
  });

  test("repairable mistakes are repaired, not discarded", () => {
    const msg = call('```json\n{"path":"a.ts",}\n```') as any;
    sanitizeToolCallArgs(msg);
    expect(JSON.parse(msg.tool_calls[0].function.arguments)).toEqual({ path: "a.ts" });
  });

  test("a JSON scalar/array is not a valid arguments object either", () => {
    for (const bad of ['"just a string"', "[1,2,3]", "42", "null"]) {
      const msg = call(bad) as any;
      expect(sanitizeToolCallArgs(msg)).toBe(true);
      const v = JSON.parse(msg.tool_calls[0].function.arguments);
      expect(v !== null && typeof v === "object" && !Array.isArray(v)).toBe(true);
    }
  });

  test("double-encoded arguments are unwrapped to a real object", () => {
    const msg = call(JSON.stringify('{"path":"a.ts"}')) as any;
    sanitizeToolCallArgs(msg);
    expect(JSON.parse(msg.tool_calls[0].function.arguments)).toEqual({ path: "a.ts" });
  });

  test("non-string / missing arguments never crash", () => {
    const a = call(undefined) as any;
    sanitizeToolCallArgs(a);
    expect(JSON.parse(a.tool_calls[0].function.arguments)).toEqual({});
    expect(sanitizeToolCallArgs({ role: "user", content: "hi" } as any)).toBe(false);
  });
});

describe("sessions are never poisoned", () => {
  test("append() stores only valid tool-call arguments", () => {
    const dir = tmpDir();
    try {
      const s = Session.create(dir, { cwd: dir, model: "m", at: "now" });
      s.append(call(POISON) as any);
      // Both in memory AND on disk — the next request reads either one.
      expect(() => JSON.parse((s.history[0] as any).tool_calls[0].function.arguments)).not.toThrow();
      const onDisk = readFileSync(join(dir, "sessions", `${s.id}.jsonl`), "utf8")
        .trim().split("\n").map((l) => JSON.parse(l)).find((o) => o.t === "msg");
      expect(() => JSON.parse(onDisk.tool_calls[0].function.arguments)).not.toThrow();
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("load() HEALS a session poisoned before the fix, and flags it recovered", () => {
    const dir = tmpDir();
    try {
      const s = Session.create(dir, { cwd: dir, model: "m", at: "now" });
      // Write the poison directly, bypassing append() — exactly what an old
      // build left on disk.
      const file = join(dir, "sessions", `${s.id}.jsonl`);
      const { appendFileSync } = require("node:fs");
      appendFileSync(file, JSON.stringify({ t: "msg", ...call(POISON) }) + "\n");
      appendFileSync(file, JSON.stringify({ t: "msg", role: "tool", tool_call_id: "call_1", content: "invalid JSON arguments" }) + "\n");

      const loaded = Session.load(dir, s.id);
      expect(loaded.recovered).toBe(true);
      const stored = (loaded.history[0] as any).tool_calls[0].function.arguments;
      expect(() => JSON.parse(stored)).not.toThrow();
      // The assistant/tool pairing must survive, or the next request 400s for
      // a different reason (dangling tool_call).
      expect(loaded.history).toHaveLength(2);
      expect((loaded.history[1] as any).tool_call_id).toBe("call_1");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("loading a clean session does not mark it recovered", () => {
    const dir = tmpDir();
    try {
      const s = Session.create(dir, { cwd: dir, model: "m", at: "now" });
      s.append(call('{"path":"a.ts"}') as any);
      expect(Session.load(dir, s.id).recovered).toBe(false);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});
