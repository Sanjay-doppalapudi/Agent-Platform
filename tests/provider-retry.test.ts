import { describe, expect, test } from "bun:test";
import { parseRetryAfterMs } from "../src/provider.ts";

describe("parseRetryAfterMs", () => {
  test("seconds", () => {
    expect(parseRetryAfterMs("5", 60_000)).toBe(5000);
    expect(parseRetryAfterMs("999", 60_000)).toBe(60_000);
  });
  test("HTTP-date", () => {
    const future = new Date(Date.now() + 10_000).toUTCString();
    const ms = parseRetryAfterMs(future, 60_000)!;
    expect(ms).toBeGreaterThan(5_000);
    expect(ms).toBeLessThanOrEqual(60_000);
  });
  test("null/garbage", () => {
    expect(parseRetryAfterMs(null)).toBeUndefined();
    expect(parseRetryAfterMs("nope")).toBeUndefined();
  });
});
