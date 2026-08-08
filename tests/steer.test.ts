import { describe, expect, test } from "bun:test";
import { drainSteerNotes, pendingSteerCount, pushSteer } from "../src/steer.ts";

describe("steer queue", () => {
  test("push → drain reaches the next turn exactly once", () => {
    drainSteerNotes(); // isolate from other tests
    expect(pushSteer("  use postgres instead  ")).toBe(true);
    expect(pendingSteerCount()).toBe(1);
    const notes = drainSteerNotes();
    expect(notes).toEqual(["use postgres instead"]);
    expect(drainSteerNotes()).toEqual([]);
    expect(pendingSteerCount()).toBe(0);
  });

  test("empty and whitespace are ignored", () => {
    drainSteerNotes();
    expect(pushSteer("")).toBe(false);
    expect(pushSteer("   \n\t  ")).toBe(false);
    expect(pendingSteerCount()).toBe(0);
  });

  test("notes are capped", () => {
    drainSteerNotes();
    pushSteer("x".repeat(10_000));
    const [n] = drainSteerNotes();
    expect(n!.length).toBe(4_000);
  });
});
