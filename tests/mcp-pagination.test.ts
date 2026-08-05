// REGRESSION: an MCP server that always returns a nextCursor made tools/list
// loop forever with no timeout — freezing the REPL before its first turn, ACP
// session/new, and `ap serve` before the port ever opened. These tests must
// TERMINATE; a regression shows up as a timeout, which is the point.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const FIXTURE = join(import.meta.dir, "fixtures", "mcp-endless.ts");

/** `ap mcp list` against a hostile server, in a throwaway workspace. */
function listTools(mode: string, timeoutMs: number) {
  const cwd = mkdtempSync(join(tmpdir(), `ap-mcp-${mode}-`));
  writeFileSync(join(cwd, ".mcp.json"), JSON.stringify({
    mcpServers: { endless: { command: process.execPath, args: ["run", FIXTURE, mode] } },
  }));
  const started = Date.now();
  const p = Bun.spawnSync(
    [process.execPath, "run", join(import.meta.dir, "..", "src", "index.ts"), "mcp", "list", "--cwd", cwd],
    { stdout: "pipe", stderr: "pipe", timeout: timeoutMs },
  );
  return {
    out: (p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? ""),
    ms: Date.now() - started,
    exitCode: p.exitCode,
  };
}

describe("MCP tools/list pagination terminates", () => {
  test("a server repeating one cursor does not hang", () => {
    const r = listTools("repeat", 60_000);
    expect(r.out).toContain("endless");
    expect(r.ms).toBeLessThan(45_000);
  }, 70_000);

  test("a server inventing a fresh cursor every page does not hang", () => {
    const r = listTools("fresh", 60_000);
    expect(r.out).toContain("endless");
    expect(r.ms).toBeLessThan(45_000);
  }, 70_000);

  test("a server returning endless EMPTY pages does not hang", () => {
    const r = listTools("empty", 60_000);
    expect(r.out).toContain("endless");
    expect(r.ms).toBeLessThan(45_000);
  }, 70_000);
});
