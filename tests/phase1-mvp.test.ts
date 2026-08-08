// Phase 1 MVP: plan/execute model resolve, cost formatting, /diff arg parsing.
import { describe, expect, test } from "bun:test";
import { providerForMode, resolveModelRef, type Config, type ResolvedProvider } from "../src/config.ts";
import { isBranchDiffArg, isWorkingDiffArg } from "../src/git.ts";
import { cacheHitPct, estimateUsd, formatUsd, type Pricing } from "../src/models.ts";

const current: ResolvedProvider = {
  name: "openrouter",
  baseUrl: "https://openrouter.ai/api/v1",
  apiKey: "k",
  model: "anthropic/claude-sonnet-4.5",
  cacheControl: false,
  headers: {},
};

const config = {
  providers: {
    openrouter: { baseUrl: "https://openrouter.ai/api/v1", model: "anthropic/claude-sonnet-4.5" },
    nim: { baseUrl: "https://nim.example/v1", model: "meta/llama" },
  },
} as unknown as Config;

describe("resolveModelRef / providerForMode", () => {
  test("bare model id stays on the current provider", () => {
    const next = resolveModelRef(config, "google/gemini-2.5-pro", current);
    expect(next.name).toBe("openrouter");
    expect(next.model).toBe("google/gemini-2.5-pro");
  });

  test("configured provider prefix switches provider", () => {
    const next = resolveModelRef(config, "nim/meta/llama-3", current);
    expect(next.name).toBe("nim");
    expect(next.model).toBe("meta/llama-3");
  });

  test("same-provider prefix strips the prefix", () => {
    const next = resolveModelRef(config, "openrouter/anthropic/claude-opus", current);
    expect(next.name).toBe("openrouter");
    expect(next.model).toBe("anthropic/claude-opus");
  });

  test("providerForMode is a no-op when unset", () => {
    expect(providerForMode(config, "plan", current)).toBe(current);
    expect(providerForMode(config, "code", current)).toBe(current);
  });

  test("providerForMode swaps when planModel/codeModel set", () => {
    const cfg = { ...config, planModel: "nim/meta/llama", codeModel: "openrouter/anthropic/claude-sonnet-4.5" } as Config;
    const plan = providerForMode(cfg, "plan", current);
    expect(plan.name).toBe("nim");
    expect(plan.model).toBe("meta/llama");
    // Resolve codeModel from the saved code snapshot (not the plan provider).
    const code = providerForMode(cfg, "code", current);
    expect(code.name).toBe("openrouter");
    expect(code.model).toBe("anthropic/claude-sonnet-4.5");
  });
});

describe("cost HUD helpers", () => {
  const pricing: Pricing = { input: 3, output: 15, cacheRead: 0.3 };
  test("estimateUsd weights cache reads cheaper", () => {
    const full = estimateUsd(pricing, { prompt: 1_000_000, cached: 0, completion: 0 });
    const cached = estimateUsd(pricing, { prompt: 1_000_000, cached: 1_000_000, completion: 0 });
    expect(full).toBe(3);
    expect(cached).toBe(0.3);
  });
  test("formatUsd and cacheHitPct", () => {
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(0.0012)).toBe("~$0.0012");
    expect(formatUsd(0.05)).toBe("~$0.050");
    expect(cacheHitPct(1000, 850)).toBe("85%");
    expect(cacheHitPct(0, 0)).toBe("");
  });
});

describe("/diff arg parsing", () => {
  test("working-tree aliases", () => {
    for (const a of ["git", "GIT", "working", "--git", "-g"]) expect(isWorkingDiffArg(a)).toBe(true);
    expect(isWorkingDiffArg("main")).toBe(false);
    expect(isWorkingDiffArg("2")).toBe(false);
  });
  test("branch refs vs checkpoint back-count", () => {
    expect(isBranchDiffArg("main")).toBe(true);
    expect(isBranchDiffArg("origin/develop")).toBe(true);
    expect(isBranchDiffArg("abc1234")).toBe(true);
    expect(isBranchDiffArg("2")).toBe(false);
    expect(isBranchDiffArg("git")).toBe(false);
    expect(isBranchDiffArg("evil;rm")).toBe(false);
  });
});
