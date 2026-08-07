import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { routeTargets, streamRouted } from "../src/router.ts";
import type { Config } from "../src/config.ts";

const config = {
  provider: "first",
  providers: {
    first: { baseUrl: "https://first.test/v1", apiKey: "one", model: "fast" },
    second: { baseUrl: "https://second.test/v1", apiKey: "two", model: "deep" },
  },
} as Config;

describe("router targets", () => {
  test("resolves namespaced provider/model targets", async () => {
    const targets = await routeTargets({ ...config, router: { targets: ["first/fast", "second/deep"] } }, {} as any);
    expect(targets.map((target) => `${target.name}/${target.model}`)).toEqual(["first/fast", "second/deep"]);
  });

  test("resolves a models.dev-only provider from catalog endpoint and env key", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "ap-catalog-"));
    writeFileSync(join(dataDir, "models-dev.json"), JSON.stringify({
      newprovider: {
        id: "newprovider",
        api: "https://newprovider.test/api",
        env: ["NEWPROVIDER_API_KEY"],
        models: { "new-model": { name: "New Model" } },
      },
    }));
    process.env.NEWPROVIDER_API_KEY = "catalog-key";
    try {
      const targets = await routeTargets({
        provider: "newprovider",
        providers: {},
        dataDir,
        router: { targets: ["newprovider/new-model"] },
      } as Config, {} as any);
      expect(targets[0]).toMatchObject({
        name: "newprovider",
        baseUrl: "https://newprovider.test/api/v1",
        apiKey: "catalog-key",
        model: "new-model",
      });
    } finally {
      delete process.env.NEWPROVIDER_API_KEY;
    }
  });

  test("uses a stored per-provider key for a models.dev-only provider", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "ap-catalog-"));
    writeFileSync(join(dataDir, "models-dev.json"), JSON.stringify({
      storedonly: {
        id: "storedonly",
        api: "https://storedonly.test",
        env: [],
        models: { "stored-model": { name: "Stored Model" } },
      },
    }));
    writeFileSync(join(dataDir, "credentials.json"), JSON.stringify({ storedonly: "stored-key" }));
    const targets = await routeTargets({
      provider: "storedonly",
      providers: {},
      dataDir,
      router: { targets: ["storedonly/stored-model"] },
    } as Config, {} as any);
    expect(targets[0]!.apiKey).toBe("stored-key");
    expect(targets[0]!.baseUrl).toBe("https://storedonly.test/v1");
  });

  test("falls back only on transient failures before output", async () => {
    const original = globalThis.fetch;
    const calls: string[] = [];
    globalThis.fetch = (async (url) => {
      calls.push(new URL(String(url)).hostname);
      if (calls.length <= 3) return new Response("busy", { status: 503 });
      return new Response("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n", {
        headers: { "content-type": "text/event-stream" },
      });
    }) as typeof fetch;
    try {
      const result = await streamRouted(
        await routeTargets({ ...config, router: { targets: ["first", "second"], fallback: true } }, {} as any),
        true,
        [{ role: "user", content: "hello" }],
        [],
        () => {},
      );
      expect(result.text).toBe("ok");
      expect(calls).toEqual(["first.test", "first.test", "first.test", "second.test"]);
    } finally {
      globalThis.fetch = original;
    }
  });

  test("does not fall back on authentication errors", async () => {
    const original = globalThis.fetch;
    const calls: string[] = [];
    globalThis.fetch = (async (url) => {
      calls.push(String(url));
      return new Response("unauthorized", { status: 401 });
    }) as typeof fetch;
    try {
      await expect(streamRouted(
        await routeTargets({ ...config, router: { targets: ["first", "second"], fallback: true } }, {} as any),
        true,
        [{ role: "user", content: "hello" }],
        [],
        () => {},
      )).rejects.toThrow("HTTP 401");
      expect(calls).toHaveLength(1);
    } finally {
      globalThis.fetch = original;
    }
  });
});
