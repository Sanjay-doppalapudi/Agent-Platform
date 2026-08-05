// ap doctor: the checks must actually FAIL on broken environments — a
// diagnostic that always says "ok" is worse than none.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { overallState, runChecks, type Check } from "../src/doctor.ts";
import { loadConfig } from "../src/config.ts";

const base = () => {
  const dir = mkdtempSync(join(tmpdir(), "ap-doctor-"));
  const c = loadConfig({ cwd: dir } as any);
  c.dataDir = mkdtempSync(join(tmpdir(), "ap-doctor-data-"));
  return c;
};
const find = (checks: Check[], name: string) => checks.find((c) => c.name === name)!;
const offline = { net: false, mcp: false };

describe("overallState", () => {
  test("worst state wins", () => {
    expect(overallState([{ name: "a", state: "ok", detail: "" }])).toBe("ok");
    expect(overallState([{ name: "a", state: "ok", detail: "" }, { name: "b", state: "warn", detail: "" }])).toBe("warn");
    expect(overallState([{ name: "a", state: "warn", detail: "" }, { name: "b", state: "fail", detail: "" }])).toBe("fail");
  });
});

describe("provider + key checks", () => {
  test("no providers configured → fail with a setup fix", async () => {
    const c = base();
    c.providers = {};
    c.provider = "";
    const checks = await runChecks(c, offline);
    const p = find(checks, "provider");
    expect(p.state).toBe("fail");
    expect(p.fix).toContain("providers");
    expect(overallState(checks)).toBe("fail");
  });

  test("provider selected but absent from the map → fail naming valid options", async () => {
    const c = base();
    c.providers = { alpha: { baseUrl: "https://a.example/v1", model: "m" } };
    c.provider = "beta";
    delete process.env.HARNESS_PROVIDER;
    const p = find(await runChecks(c, offline), "provider");
    expect(p.state).toBe("fail");
    expect(p.detail).toContain("beta");
    expect(p.fix).toContain("alpha");
  });

  test("provider without a model → warn (usable with -m)", async () => {
    const c = base();
    c.providers = { alpha: { baseUrl: "https://a.example/v1", model: "" } };
    c.provider = "alpha";
    expect(find(await runChecks(c, offline), "provider").state).toBe("warn");
  });

  test("missing key → fail telling you the exact auth command", async () => {
    const c = base();
    c.providers = { alpha: { baseUrl: "https://a.example/v1", model: "m", apiKeyEnv: "NO_SUCH_KEY_VAR" } };
    c.provider = "alpha";
    delete process.env.HARNESS_API_KEY;
    delete process.env.HARNESS_PROVIDER;
    const k = find(await runChecks(c, offline), "api key");
    expect(k.state).toBe("fail");
    expect(k.fix).toContain("ap auth alpha");
    expect(k.fix).toContain("NO_SUCH_KEY_VAR");
  });

  test("key present in config → ok", async () => {
    const c = base();
    c.providers = { alpha: { baseUrl: "https://a.example/v1", model: "m", apiKey: "sk-test" } };
    c.provider = "alpha";
    delete process.env.HARNESS_PROVIDER;
    expect(find(await runChecks(c, offline), "api key").state).toBe("ok");
  });
});

describe("credential store", () => {
  test("corrupt credentials.json → fail with a recovery step", async () => {
    const c = base();
    writeFileSync(join(c.dataDir, "credentials.json"), "{ not json");
    const cred = find(await runChecks(c, offline), "credentials");
    expect(cred.state).toBe("fail");
    expect(cred.fix).toContain("ap auth");
  });

  test("absent store is fine (env/config keys are valid)", async () => {
    expect(find(await runChecks(base(), offline), "credentials").state).toBe("ok");
  });
});

describe("environment checks", () => {
  test("ripgrep and git are probed and reported", async () => {
    const checks = await runChecks(base(), offline);
    expect(["ok", "fail"]).toContain(find(checks, "ripgrep").state);
    expect(["ok", "warn"]).toContain(find(checks, "git").state);
  });

  test("data dir writability is proven by an actual write", async () => {
    expect(find(await runChecks(base(), offline), "data dir").state).toBe("ok");
  });

  test("every non-ok check carries an actionable fix", async () => {
    const c = base();
    c.providers = {};
    c.provider = "";
    for (const chk of await runChecks(c, offline)) {
      if (chk.state !== "ok") expect(chk.fix, `${chk.name} has no fix`).toBeTruthy();
    }
  });

  test("offline mode skips the network and MCP probes", async () => {
    const names = (await runChecks(base(), offline)).map((c) => c.name);
    expect(names).not.toContain("endpoint");
    expect(names).not.toContain("mcp");
  });
});
