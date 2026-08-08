// Tool registry: aliases, schema sets, arg validation, permission rules.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import {
  execTool, isParallelSafe, permissionFor, resolveToolName, toolSchemasFor,
} from "../src/tools/index.ts";

const tmp = mkdtempSync(join(tmpdir(), "ap-test-"));
const config = loadConfig({ cwd: tmp } as any);
const ctx = { cwd: tmp, signal: new AbortController().signal, config, permit: async () => true };

describe("resolveToolName", () => {
  test("exact names pass through", () => expect(resolveToolName("grep")).toBe("grep"));
  test("aliases map to canonical", () => {
    expect(resolveToolName("search")).toBe("grep");
    expect(resolveToolName("create_file")).toBe("write");
    expect(resolveToolName("str_replace_editor")).toBe("edit");
    expect(resolveToolName("cat")).toBe("read");
    expect(resolveToolName("shell")).toBe("bash");
    expect(resolveToolName("web_search")).toBe("websearch");
  });
  test("unknown names pass through unchanged", () => expect(resolveToolName("zzz")).toBe("zzz"));
});

describe("toolSchemasFor", () => {
  const names = (c: any) => toolSchemasFor(c).map((s: any) => s.function.name);
  test("light profile is exactly the core six", () => {
    expect(names({ ...config, light: true, mode: "code" })).toEqual(
      ["read", "write", "edit", "bash", "glob", "grep"],
    );
  });
  test("plan mode has no mutating tools", () => {
    const plan = names({ ...config, mode: "plan" });
    for (const n of ["write", "edit", "bash", "agent"]) expect(plan).not.toContain(n);
    expect(plan).toContain("read");
  });
  test("toolFilter whitelists (agent profiles)", () => {
    const filtered = names({ ...config, mode: "code", toolFilter: ["read", "grep"] });
    expect(filtered.sort()).toEqual(["grep", "read"]);
  });
  test("bogus toolFilter falls back to the full set", () => {
    expect(names({ ...config, mode: "code", toolFilter: ["nope"] }).length).toBeGreaterThan(5);
  });
});

describe("isParallelSafe", () => {
  test("read-only tools are parallel-safe", () => expect(isParallelSafe("grep")).toBe(true));
  test("mutating tools are serialized", () => expect(isParallelSafe("edit")).toBe(false));
  test("aliases are resolved first", () => expect(isParallelSafe("search")).toBe(true));
});

describe("execTool", () => {
  test("missing required argument fails fast with guidance", async () => {
    const r = await execTool("read", "{}", ctx as any);
    expect(r.error).toBe(true);
    expect(r.output).toContain("missing required argument");
    expect(r.output).toContain("path");
  });
  test("unknown tool names the available set", async () => {
    const r = await execTool("teleport", "{}", ctx as any);
    expect(r.error).toBe(true);
    expect(r.output).toContain("unknown tool");
  });
  test("lenient JSON: trailing commas + arg aliases still work", async () => {
    writeFileSync(join(tmp, "fixture.txt"), "hello\nworld\n");
    const r = await execTool("read", '{"file_path": "fixture.txt",}', ctx as any);
    expect(r.error).toBe(false);
    expect(r.output).toContain("hello");
  });
});

describe("permissionFor", () => {
  const cfg: any = {
    permission: {
      fetch: "deny",
      edit: "ask",
      "mcp_*": "ask",
      bash: { "git push*": "ask", "rm *": "deny", "*": "allow" },
    },
  };
  test("exact deny", () => expect(permissionFor(cfg, "fetch", "{}")).toBe("deny"));
  test("glob over tool names", () => expect(permissionFor(cfg, "mcp_github_x", "{}")).toBe("ask"));
  test("bash command patterns", () => {
    expect(permissionFor(cfg, "bash", JSON.stringify({ cmd: "git push origin" }))).toBe("ask");
    expect(permissionFor(cfg, "bash", JSON.stringify({ cmd: "rm -rf build" }))).toBe("deny");
    expect(permissionFor(cfg, "bash", JSON.stringify({ cmd: "echo hi" }))).toBe("allow");
    expect(permissionFor(cfg, "bash", JSON.stringify({ command: "git push" }))).toBe("ask"); // arg alias
    // Compound: strictest segment wins
    expect(permissionFor(cfg, "bash", JSON.stringify({ cmd: "echo hi && git push origin main" }))).toBe("ask");
    expect(permissionFor(cfg, "bash", JSON.stringify({ cmd: "echo hi; rm -rf build" }))).toBe("deny");
  });
  test("no rule → null (falls back to permissions mode)", () => {
    expect(permissionFor(cfg, "read", "{}")).toBeNull();
    expect(permissionFor({} as any, "bash", "{}")).toBeNull();
  });

  // SECURITY REGRESSION: permissionFor used strict JSON.parse while execTool
  // used the lenient one, so malformed-but-repairable arguments evaded deny
  // rules and then ran anyway. Every shape execTool can repair must be
  // evaluated by the SAME rules.
  describe("malformed arguments cannot evade a deny rule", () => {
    const deny: any = { permission: { bash: { "rm *": "deny", "*": "allow" } } };
    const variants: [string, string][] = [
      ["strict json", '{"cmd":"rm -rf build"}'],
      ["trailing comma", '{"cmd": "rm -rf build",}'],
      ["fenced", '```json\n{"cmd": "rm -rf build"}\n```'],
      ["double-encoded", JSON.stringify('{"cmd": "rm -rf build"}')],
      ["smart quotes", '{“cmd”: “rm -rf build”}'],
    ];
    for (const [label, args] of variants) {
      test(`${label} → deny`, () => expect(permissionFor(deny, "bash", args)).toBe("deny"));
    }
    test("allow rules still work on repaired args", () => {
      expect(permissionFor(deny, "bash", '{"cmd": "echo hi",}')).toBe("allow");
    });
    test("truly unparseable args fail closed when a strict rule exists", () => {
      expect(permissionFor(deny, "bash", "{not json at all")).toBe("ask");
    });
    test("truly unparseable args use the default when nothing is strict", () => {
      const permissive: any = { permission: { bash: { "*": "allow" } } };
      expect(permissionFor(permissive, "bash", "{not json at all")).toBe("allow");
    });
  });
});

describe("parseArgsLenient double-encoding (via execTool)", () => {
  test("double-encoded arguments are decoded, not dropped", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ap-test-dbl-"));
    writeFileSync(join(dir, "f.txt"), "content here\n");
    const c = loadConfig({ cwd: dir } as any);
    const r = await execTool("read", JSON.stringify(JSON.stringify({ path: "f.txt" })), {
      cwd: dir, signal: new AbortController().signal, config: c, permit: async () => true,
    } as any);
    expect(r.error).toBe(false);
    expect(r.output).toContain("content here");
  });
});

describe("grep file path (Windows cwd)", () => {
  test("grepping a single file does not set cwd to that file", async () => {
    writeFileSync(join(tmp, "needle.txt"), "hello Agent Platform world\n");
    const r = await execTool(
      "grep",
      JSON.stringify({ pattern: "Agent Platform", path: join(tmp, "needle.txt") }),
      ctx,
    );
    expect(r.error).toBeFalsy();
    expect(r.output).toContain("Agent Platform");
  });
});
