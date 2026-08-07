// ap serve: production routes bind loopback by default, require a bearer
// token when configured, and keep provider/model out of the public health body.
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { serveMain } from "../src/server.ts";

async function startTestServe(token = "sekret") {
  const cwd = mkdtempSync(join(tmpdir(), "ap-serve-auth-"));
  const dataDir = join(cwd, "data");
  mkdirSync(dataDir);
  // Keep server state isolated, disable any user-level MCP configuration, and
  // inject a local ad-hoc provider that is never contacted in these route tests.
  writeFileSync(join(cwd, "ap.config.json"), JSON.stringify({ dataDir, mcpServers: {} }));
  const server = await serveMain({
    cwd,
    baseUrl: "http://127.0.0.1:9/v1",
    apiKey: "test-key",
    model: "test-model",
    port: 0,
    host: "127.0.0.1",
    token,
  });
  return { server, url: `http://127.0.0.1:${server.port}` };
}

describe("serve auth model", () => {
  test("health is public but does not advertise provider/model", async () => {
    const srv = await startTestServe();
    try {
      const r = await fetch(`${srv.url}/health`);
      const body = await r.json() as any;
      expect(r.status).toBe(200);
      expect(body.ok).toBe(true);
      expect(body.provider).toBeUndefined();
      expect(body.model).toBeUndefined();
      expect(body.auth).toBe("required");
    } finally { srv.server.stop(true); }
  });

  test("session routes require the bearer token", async () => {
    const srv = await startTestServe();
    try {
      const denied = await fetch(`${srv.url}/session`, { method: "POST" });
      expect(denied.status).toBe(401);
      const ok = await fetch(`${srv.url}/session`, {
        method: "POST",
        headers: { authorization: "Bearer sekret", "content-type": "application/json" },
      });
      expect(ok.status).toBe(200);
      expect((await ok.json() as any).id).toBeString();
    } finally { srv.server.stop(true); }
  });

  test("?token= works for SSE-style clients", async () => {
    const srv = await startTestServe();
    try {
      const r = await fetch(`${srv.url}/session?token=sekret`, { method: "POST" });
      expect(r.status).toBe(200);
    } finally { srv.server.stop(true); }
  });
});

describe("serve defaults", () => {
  test("CliFlags accept --host and --token", async () => {
    // parseArgs is not exported; exercise via the binary's --help topic instead
    // and confirm the source contract by reading the flag names from a dry run.
    const cwd = mkdtempSync(join(tmpdir(), "ap-serve-flags-"));
    const proc = Bun.spawn(
      [process.execPath, join(import.meta.dir, "../src/index.ts"), "help", "serve"],
      { cwd, stdout: "pipe", stderr: "pipe", env: { ...process.env } },
    );
    const out = await new Response(proc.stdout).text();
    await proc.exited;
    expect(out).toContain("--host");
    expect(out).toContain("--token");
    expect(out).toMatch(/127\.0\.0\.1/);
  });
});
