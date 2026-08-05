// REGRESSION: `ap doctor` reported "all checks passed" while every turn died
// with HTTP 401. It only checked that a key RESOLVED, never that the provider
// accepts it — a false OK, the one answer a diagnostic must never give.
import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { overallState, runChecks, type Check } from "../src/doctor.ts";

/** A stand-in provider that answers with a fixed status. */
function fakeProvider(status: number, body: unknown) {
  return Bun.serve({
    port: 0,
    fetch: () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }),
  });
}

function configFor(port: number) {
  const cwd = mkdtempSync(join(tmpdir(), "ap-doctor-auth-"));
  const config = loadConfig({ cwd } as any);
  config.dataDir = mkdtempSync(join(tmpdir(), "ap-doctor-auth-data-"));
  config.provider = "fake";
  config.providers = { fake: { baseUrl: `http://localhost:${port}/v1`, model: "test-model", apiKey: "sk-test" } };
  delete process.env.HARNESS_PROVIDER;
  delete process.env.HARNESS_API_KEY;
  return config;
}
const auth = (checks: Check[]) => checks.find((c) => c.name === "auth")!;

describe("doctor validates the key against the provider", () => {
  test("a rejected key FAILS the run (this is the reported bug)", async () => {
    const srv = fakeProvider(401, { type: "error", error: { type: "AuthError", message: "Request blocked by upstream provider." } });
    try {
      const checks = await runChecks(configFor(srv.port), { mcp: false });
      const a = auth(checks);
      expect(a.state).toBe("fail");
      expect(a.detail).toContain("REJECTED");
      expect(a.detail).toContain("401");
      expect(a.fix).toMatch(/revoked|credits|blocked/i);
      expect(overallState(checks)).toBe("fail"); // must NOT report "all checks passed"
    } finally { srv.stop(true); }
  });

  test("403 is also a failure", async () => {
    const srv = fakeProvider(403, { error: "forbidden" });
    try {
      expect(auth(await runChecks(configFor(srv.port), { mcp: false })).state).toBe("fail");
    } finally { srv.stop(true); }
  });

  test("a working key passes", async () => {
    const srv = fakeProvider(200, { choices: [{ message: { content: "hi" } }] });
    try {
      const checks = await runChecks(configFor(srv.port), { mcp: false });
      expect(auth(checks).state).toBe("ok");
      expect(auth(checks).detail).toContain("accepted the key");
      expect(overallState(checks)).not.toBe("fail");
    } finally { srv.stop(true); }
  });

  test("rate limiting is a warning, not a failure — the key is valid", async () => {
    const srv = fakeProvider(429, { error: "slow down" });
    try {
      expect(auth(await runChecks(configFor(srv.port), { mcp: false })).state).toBe("warn");
    } finally { srv.stop(true); }
  });

  test("404 points at the model id / base URL, not the key", async () => {
    const srv = fakeProvider(404, { error: "no such model" });
    try {
      const a = auth(await runChecks(configFor(srv.port), { mcp: false }));
      expect(a.state).toBe("fail");
      expect(a.fix).toMatch(/model id|base URL/i);
    } finally { srv.stop(true); }
  });

  test("the auth probe is skipped offline", async () => {
    const srv = fakeProvider(401, {});
    try {
      const names = (await runChecks(configFor(srv.port), { net: false, mcp: false })).map((c) => c.name);
      expect(names).not.toContain("auth");
    } finally { srv.stop(true); }
  });
});

describe("errorHint distinguishes rejected from missing keys", () => {
  test("a blocked/revoked key does not say 'missing'", async () => {
    const { errorHint } = await import("../src/ui.ts");
    const hint = errorHint('HTTP 401: {"error":{"type":"AuthError","message":"Request blocked by upstream provider."}}')!;
    expect(hint).toMatch(/REJECTED/);
    expect(hint).not.toMatch(/missing or wrong/);
  });
  test("a genuinely absent key still points at ap auth", async () => {
    const { errorHint } = await import("../src/ui.ts");
    expect(errorHint("no api key for provider \"x\"")).toContain("ap auth");
  });
});
