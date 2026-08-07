// /model interactive picker: the pure row builders (provider merge/badges/
// sort, model hints) and the raw-mode list selector driven by synthetic
// keypresses — same technique as vt.test.ts, because selection/filter/cancel
// behavior is not observable from pure functions.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { emitKeypressEvents } from "node:readline";
import { buildModelRows, buildProviderRows, pickFromList } from "../src/picker.ts";
import { setKey } from "../src/creds.ts";
import { setTheme } from "../src/theme.ts";
import type { Catalog } from "../src/models.ts";
import type { Config } from "../src/config.ts";

setTheme("mono");
emitKeypressEvents(process.stdin);

const savedHarnessKey = process.env.HARNESS_API_KEY;
beforeAll(() => { delete process.env.HARNESS_API_KEY; }); // overrides every badge
afterAll(() => { if (savedHarnessKey !== undefined) process.env.HARNESS_API_KEY = savedHarnessKey; });

const CATALOG: Catalog = {
  alpha: { id: "alpha", name: "Alpha AI", api: "https://api.alpha.test", env: ["AP_PICKER_ALPHA_KEY"], models: { "a-1": {}, "a-2": {} } },
  beta: { id: "beta", api: "https://api.beta.test/v1", env: ["AP_PICKER_BETA_KEY"], models: { "b-1": { cost: { input: 1, output: 2 }, limit: { context: 200_000 } } } },
  nolisting: { id: "nolisting", models: { "x-1": {} } }, // no api → unusable
};

function fakeConfig(dataDir: string, providers: Config["providers"] = {}): Config {
  return { dataDir, providers } as Config;
}

describe("buildProviderRows", () => {
  const dir = mkdtempSync(join(tmpdir(), "ap-picker-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("catalog providers without an api endpoint are excluded; configured ones always surface", () => {
    const rows = buildProviderRows(fakeConfig(dir, { custom: { baseUrl: "https://my.test/v1", model: "m1" } }), CATALOG);
    const ids = rows.map((r) => r.value);
    expect(ids).toContain("alpha");
    expect(ids).toContain("beta");
    expect(ids).toContain("custom");
    expect(ids).not.toContain("nolisting");
  });

  test("key badges follow real resolution: env var, credential store, none", () => {
    process.env.AP_PICKER_ALPHA_KEY = "sk-env";
    try {
      const cfg = fakeConfig(dir);
      setKey(dir, "beta", "sk-stored");
      const rows = buildProviderRows(cfg, CATALOG);
      const hint = (id: string) => rows.find((r) => r.value === id)!.hint!;
      expect(hint("alpha")).toContain("✓ key"); // env var
      expect(hint("beta")).toContain("✓ key"); // credential store
    } finally { delete process.env.AP_PICKER_ALPHA_KEY; }
  });

  test("keyed providers sort first; a configured baseUrl ignores catalog env (resolution mirror)", () => {
    process.env.AP_PICKER_ALPHA_KEY = "sk-env";
    try {
      // alpha configured WITH baseUrl → resolution never consults the catalog
      // env list, so the badge must not claim a key it won't use.
      const cfg = fakeConfig(mkdtempSync(join(tmpdir(), "ap-picker2-")), {
        alpha: { baseUrl: "https://pinned.test/v1", model: "a-1", apiKeyEnv: "AP_PICKER_UNSET_ENV" },
      });
      const rows = buildProviderRows(cfg, CATALOG);
      const alpha = rows.find((r) => r.value === "alpha")!;
      expect(alpha.hint).toContain("needs key");
      // beta (no key) and alpha (configured, no key): both unkeyed → configured wins the tie.
      expect(rows.findIndex((r) => r.value === "alpha")).toBeLessThan(rows.findIndex((r) => r.value === "beta"));
    } finally { delete process.env.AP_PICKER_ALPHA_KEY; }
  });

  test("model counts include the configured-default fallback", () => {
    const rows = buildProviderRows(fakeConfig(dir, { custom: { baseUrl: "https://my.test/v1", model: "m1" } }), CATALOG);
    expect(rows.find((r) => r.value === "custom")!.hint).toContain("1 model");
    expect(rows.find((r) => r.value === "alpha")!.hint).toContain("2 models");
  });
});

describe("buildModelRows", () => {
  test("catalog models get ctx and cost hints, sorted by id", () => {
    const rows = buildModelRows(CATALOG.beta);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.label).toBe("b-1");
    expect(rows[0]!.hint).toBe("200k · $1/$2");
  });
  test("no catalog listing falls back to the configured default", () => {
    const rows = buildModelRows(undefined, "my-model");
    expect(rows).toEqual([{ label: "my-model", hint: "configured default", value: "my-model" }]);
  });
  test("nothing at all → empty", () => {
    expect(buildModelRows(undefined, undefined)).toEqual([]);
  });
});

describe("pickFromList", () => {
  /** Run a picker with stdout swallowed, feed keys, return the resolution. */
  async function drive<T>(items: { label: string; hint?: string; value: T }[], keys: { str?: string; key: any }[], initial = 0): Promise<T | null> {
    const realWrite = process.stdout.write.bind(process.stdout);
    (process.stdout as any).write = () => true;
    try {
      const p = pickFromList("test", items, initial);
      for (const k of keys) process.stdin.emit("keypress", k.str ?? "", k.key);
      return await p;
    } finally { (process.stdout as any).write = realWrite; }
  }
  const items = [
    { label: "openai", value: "openai" },
    { label: "openrouter", value: "openrouter" },
    { label: "groq", value: "groq" },
  ];

  test("Enter picks the highlighted item; ↓ moves the highlight", async () => {
    expect(await drive(items, [{ key: { name: "down" } }, { key: { name: "return" } }])).toBe("openrouter");
  });

  test("typing filters, Enter takes the first match", async () => {
    expect(await drive(items, [
      { str: "g", key: { name: "g" } }, { str: "r", key: { name: "r" } }, { str: "o", key: { name: "o" } }, { str: "q", key: { name: "q" } },
      { key: { name: "return" } },
    ])).toBe("groq");
  });

  test("filter resets the highlight to the top (no stale index carryover)", async () => {
    expect(await drive(items, [
      { key: { name: "down" } }, { key: { name: "down" } },
      { str: "o", key: { name: "o" } }, // "o" matches openai, openrouter, groq — idx back to 0
      { key: { name: "return" } },
    ])).toBe("openai");
  });

  test("Esc cancels with null; ctrl+c cancels without killing the process", async () => {
    expect(await drive(items, [{ key: { name: "escape" } }])).toBeNull();
    expect(await drive(items, [{ key: { name: "c", ctrl: true } }])).toBeNull();
  });

  test("initial index preselects (current provider opens highlighted)", async () => {
    expect(await drive(items, [{ key: { name: "return" } }], 2)).toBe("groq");
  });

  test("Enter on an empty filter result is a cancel, not a crash", async () => {
    expect(await drive(items, [
      { str: "z", key: { name: "z" } }, { str: "z", key: { name: "z" } },
      { key: { name: "return" } },
    ])).toBeNull();
  });
});
