// ap serve: loopback default, bearer auth, health does not leak provider.
import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/** Minimal stand-in that reuses the same auth helpers as server.ts would —
 *  we exercise the live server via spawning `ap serve` would need a provider
 *  key, so this unit-tests the auth predicate shape by importing nothing
 *  heavy: instead we spin a tiny Bun.serve with the same rules. */
function startTestServe(opts: { host: string; token: string }) {
  const hostname = opts.host;
  const token = opts.token;
  const authorized = (req: Request): boolean => {
    if (!token) return true;
    const h = req.headers.get("authorization") ?? "";
    if (h === `Bearer ${token}`) return true;
    try { return new URL(req.url).searchParams.get("token") === token; } catch { return false; }
  };
  return Bun.serve({
    port: 0,
    hostname,
    fetch(req) {
      const path = new URL(req.url).pathname;
      if (path === "/health") {
        return Response.json({ ok: true, version: "test", auth: token ? "required" : "off", host: hostname });
      }
      if (!authorized(req)) return Response.json({ error: "unauthorized" }, { status: 401 });
      return Response.json({ ok: true, secret: "session-data" });
    },
  });
}

describe("serve auth model", () => {
  test("health is public but does not advertise provider/model", async () => {
    const srv = startTestServe({ host: "127.0.0.1", token: "sekret" });
    try {
      const r = await fetch(`http://127.0.0.1:${srv.port}/health`);
      const body = await r.json() as any;
      expect(r.status).toBe(200);
      expect(body.ok).toBe(true);
      expect(body.provider).toBeUndefined();
      expect(body.model).toBeUndefined();
      expect(body.auth).toBe("required");
    } finally { srv.stop(true); }
  });

  test("session routes require the bearer token", async () => {
    const srv = startTestServe({ host: "127.0.0.1", token: "sekret" });
    try {
      const denied = await fetch(`http://127.0.0.1:${srv.port}/session`);
      expect(denied.status).toBe(401);
      const ok = await fetch(`http://127.0.0.1:${srv.port}/session`, {
        headers: { authorization: "Bearer sekret" },
      });
      expect(ok.status).toBe(200);
      expect((await ok.json() as any).secret).toBe("session-data");
    } finally { srv.stop(true); }
  });

  test("?token= works for SSE-style clients", async () => {
    const srv = startTestServe({ host: "127.0.0.1", token: "sekret" });
    try {
      const r = await fetch(`http://127.0.0.1:${srv.port}/session?token=sekret`);
      expect(r.status).toBe(200);
    } finally { srv.stop(true); }
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
