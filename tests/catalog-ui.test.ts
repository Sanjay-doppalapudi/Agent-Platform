// models.dev helpers (fake catalog — no network) + UI pure functions.
import { describe, expect, test } from "bun:test";
import { modelPricing, modelReasoning, providerBaseUrl, searchModels, type Catalog } from "../src/models.ts";
import { errorHint, renderDiff, toolLabel } from "../src/ui.ts";

const catalog: Catalog = {
  acme: {
    id: "acme",
    api: "https://api.acme.dev",
    models: {
      "fast-1": { cost: { input: 1, output: 2, cache_read: 0.1 }, reasoning: false },
      "think-1": { cost: { input: 3, output: 15 }, reasoning: true, limit: { context: 200000 } },
    },
  },
  gateway: { id: "gateway", models: {} },
};

describe("modelPricing", () => {
  test("exact provider hit", () => {
    expect(modelPricing(catalog, "acme", "fast-1")).toEqual({ input: 1, output: 2, cacheRead: 0.1 });
  });
  test("gateway falls back to any provider listing the id", () => {
    expect(modelPricing(catalog, "gateway", "think-1")?.input).toBe(3);
  });
  test("proxy-style prefixed ids resolve via the short name", () => {
    expect(modelPricing(catalog, "gateway", "acme/fast-1")?.input).toBe(1);
  });
  test("unknown → null", () => expect(modelPricing(catalog, "x", "nope")).toBeNull());
});

describe("modelReasoning", () => {
  test("true/false from catalog", () => {
    expect(modelReasoning(catalog, "acme", "think-1")).toBe(true);
    expect(modelReasoning(catalog, "acme", "fast-1")).toBe(false);
  });
  test("unknown → null", () => expect(modelReasoning(catalog, "x", "nope")).toBeNull());
});

describe("providerBaseUrl", () => {
  test("appends /v1 when missing", () => {
    expect(providerBaseUrl(catalog["acme"]!)).toBe("https://api.acme.dev/v1");
  });
  test("keeps an existing version segment", () => {
    expect(providerBaseUrl({ id: "x", api: "https://a.b/v1", models: {} })).toBe("https://a.b/v1");
  });
});

describe("searchModels", () => {
  test("substring match over provider/model", () => {
    expect(searchModels(catalog, "think").map((r) => r.model)).toEqual(["think-1"]);
  });
});

describe("errorHint", () => {
  test("model errors outrank auth errors (401-wrapped model error)", () => {
    expect(errorHint('HTTP 401: {"message":"Model x is not supported"}')).toContain("/model");
  });
  test("plain 401 → auth hint", () => {
    expect(errorHint("HTTP 401 unauthorized")).toContain("ap auth");
  });
  test("rate limit → note", () => expect(errorHint("429 rate limit exceeded")).toContain("rate-limited"));
  test("unknown error → null", () => expect(errorHint("something exploded")).toBeNull());
});

describe("toolLabel / renderDiff", () => {
  test("labels are plain text", () => {
    expect(toolLabel("read", { path: "a.ts", offset: 3 })).toBe("read a.ts:3");
    expect(toolLabel("bash", { cmd: "echo   hi", background: true })).toContain("&");
  });
  test("edit diff trims common context and marks +/-", () => {
    const d = renderDiff("edit", { path: "x.ts", old: "a\nb\nc", new: "a\nB\nc" }, 80);
    expect(d).toContain("- b");
    expect(d).toContain("+ B");
    expect(d).not.toContain("- a"); // common prefix trimmed
  });
  test("edit with hallucinated empty args renders nothing", () => {
    expect(renderDiff("edit", {})).toBe("");
  });
});
