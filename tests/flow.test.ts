// Workflows: path resolution (never a traversal vector), the concurrency-
// capped parallel, JSON extraction, and the agent() schema/retry contract —
// driven with an injected runner, no child processes.
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { boundedParallel, extractJson, resolveFlowPath, runFlow, validateShape } from "../src/flow.ts";

function tmpConfig() {
  const cwd = mkdtempSync(join(tmpdir(), "ap-flow-cwd-"));
  const dataDir = mkdtempSync(join(tmpdir(), "ap-flow-data-"));
  return { config: { cwd, dataDir } as any, cleanup: () => { rmSync(cwd, { recursive: true, force: true }); rmSync(dataDir, { recursive: true, force: true }); } };
}

describe("resolveFlowPath", () => {
  test(".ap/workflows/<name>.ts wins; names must be identifiers", () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "audit.ts"), "export default async () => 1");
      expect(resolveFlowPath(config, "audit")).toBe(join(dir, "audit.ts"));
      expect(resolveFlowPath(config, "missing")).toBeNull();
      // A name is an identifier, not a path — traversal shapes resolve to null.
      expect(resolveFlowPath(config, "../../etc/passwd")).toBeNull();
      expect(resolveFlowPath(config, "..\\..\\x")).toBeNull();
    } finally { cleanup(); }
  });

  test("explicit .ts path is allowed (it is the user typing a path)", () => {
    const { config, cleanup } = tmpConfig();
    try {
      const p = join(config.cwd, "myflow.ts");
      writeFileSync(p, "export default async () => 1");
      expect(resolveFlowPath(config, "myflow.ts")).toBe(p);
      expect(resolveFlowPath(config, "nope.ts")).toBeNull();
    } finally { cleanup(); }
  });
});

describe("boundedParallel", () => {
  test("never exceeds the cap and preserves order", async () => {
    let live = 0;
    let peak = 0;
    const thunk = (n: number) => async () => {
      live++; peak = Math.max(peak, live);
      await new Promise((r) => setTimeout(r, 10));
      live--;
      return n;
    };
    const out = await boundedParallel(Array.from({ length: 12 }, (_, i) => thunk(i)), 3);
    expect(out).toEqual(Array.from({ length: 12 }, (_, i) => i));
    expect(peak).toBeLessThanOrEqual(3);
  });

  test("a failing thunk becomes null, the rest complete", async () => {
    const out = await boundedParallel([
      async () => "ok",
      async () => { throw new Error("boom"); },
      async () => "also ok",
    ], 2);
    expect(out).toEqual(["ok", null, "also ok"]);
  });
});

describe("extractJson", () => {
  test("plain JSON, fenced JSON, and JSON with prose around it", () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 });
    expect(extractJson('```json\n{"a":1}\n```')).toEqual({ a: 1 });
    expect(extractJson('Sure! Here it is: {"a":{"b":[1,2]}} hope that helps')).toEqual({ a: { b: [1, 2] } });
    expect(extractJson('text ["x","y{"] more')).toEqual(["x", "y{"]); // brace inside a string
  });
  test("no JSON → throws", () => {
    expect(() => extractJson("no json here")).toThrow();
  });
});

describe("runFlow with an injected runner", () => {
  test("flow gets agent/parallel/log/args; result comes back", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "hello.ts"), `
        export default async function ({ agent, parallel, log, args }) {
          log("starting");
          const one = await agent("say one");
          const many = await parallel([() => agent("a"), () => agent("b")]);
          return { one, many, args };
        }`);
      const calls: string[] = [];
      const lines: string[] = [];
      const result = await runFlow(config, "hello", ["x"], (l) => lines.push(l), async (task) => {
        calls.push(task);
        return `reply:${task}`;
      });
      expect(result).toEqual({ one: "reply:say one", many: ["reply:a", "reply:b"], args: ["x"] });
      expect(calls).toContain("say one");
      expect(lines).toContain("starting");
    } finally { cleanup(); }
  });

  test("schema: invalid JSON gets ONE retry with the error fed back, then parses", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "schema.ts"), `
        export default async function ({ agent }) {
          return agent("classify", { schema: { type: "object" } });
        }`);
      let n = 0;
      const result = await runFlow(config, "schema", [], () => {}, async (task) => {
        n++;
        if (n === 1) return "not json at all";
        expect(task).toContain("was rejected");
        expect(task).toContain("no JSON found");
        return '{"label":"bug"}';
      });
      expect(n).toBe(2);
      expect(result).toEqual({ label: "bug" });
    } finally { cleanup(); }
  });

  test("the agent cap stops runaway flows", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "runaway.ts"), `
        export default async function ({ agent }) {
          for (;;) await agent("again");
        }`);
      await expect(runFlow(config, "runaway", [], () => {}, async () => "ok")).rejects.toThrow(/exceeded 50 agent calls/);
    } finally { cleanup(); }
  });

  test("a flow without a default function export is rejected", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "bad.ts"), `export const x = 1`);
      await expect(runFlow(config, "bad", [], () => {})).rejects.toThrow(/export default async function/);
    } finally { cleanup(); }
  });
});

describe("validateShape (AUDIT: the schema contract promised shape checking but only parsed JSON)", () => {
  const S = {
    type: "object",
    required: ["findings"],
    properties: {
      findings: {
        type: "array",
        items: { type: "object", required: ["title"], properties: { title: { type: "string" }, n: { type: "integer" } } },
      },
    },
  };
  test("conforming values pass", () => {
    expect(validateShape({ findings: [{ title: "x", n: 1 }] }, S)).toBeNull();
    expect(validateShape({ findings: [] }, S)).toBeNull();
  });
  test("wrong top-level type is named", () => {
    expect(validateShape([], S)).toContain("should be object");
    expect(validateShape("nope", S)).toContain("should be object");
  });
  test("missing required key is named with its path", () => {
    expect(validateShape({}, S)).toBe("value.findings is required but missing");
  });
  test("nested array item errors carry an index path", () => {
    expect(validateShape({ findings: [{ title: "ok" }, {}] }, S)).toBe("value.findings[1].title is required but missing");
    expect(validateShape({ findings: [{ title: 5 }] }, S)).toContain("value.findings[0].title should be string");
  });
  test("scalar where an array belongs", () => {
    expect(validateShape({ findings: "x" }, S)).toContain("value.findings should be array");
  });
  test("integer vs number, and enum membership", () => {
    expect(validateShape(1.5, { type: "integer" })).toContain("should be integer");
    expect(validateShape(2, { type: "integer" })).toBeNull();
    expect(validateShape("c", { enum: ["a", "b"] })).toContain("should be one of");
  });
  test("an empty/absent schema accepts anything", () => {
    expect(validateShape({ any: "thing" }, {})).toBeNull();
    expect(validateShape(null, undefined)).toBeNull();
  });
});

describe("agent(schema) enforces the shape, not just parseability", () => {
  test("shape-invalid JSON triggers the retry with the field path fed back", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "shaped.ts"), `
        export default async function ({ agent }) {
          return agent("classify", { schema: { type: "object", required: ["label"], properties: { label: { type: "string" } } } });
        }`);
      let n = 0;
      let retryPrompt = "";
      const result = await runFlow(config, "shaped", [], () => {}, async (task) => {
        n++;
        if (n === 1) return '{"wrong":"key"}'; // parses fine, wrong SHAPE
        retryPrompt = task;
        return '{"label":"bug"}';
      });
      expect(n).toBe(2);
      expect(retryPrompt).toContain("label is required but missing");
      expect(result).toEqual({ label: "bug" });
    } finally { cleanup(); }
  });

  test("a second shape failure throws to the flow rather than looping", async () => {
    const { config, cleanup } = tmpConfig();
    try {
      const dir = join(config.cwd, ".ap", "workflows");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "never.ts"), `
        export default async function ({ agent }) {
          return agent("x", { schema: { type: "object", required: ["a"] } });
        }`);
      let n = 0;
      await expect(runFlow(config, "never", [], () => {}, async () => { n++; return "{}"; }))
        .rejects.toThrow(/schema mismatch/);
      expect(n).toBe(2); // exactly one retry, then give up
    } finally { cleanup(); }
  });
});
