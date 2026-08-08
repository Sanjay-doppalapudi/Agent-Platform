import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { drainChannelNotes, postChannel, readChannel, safeChannelId } from "../src/channels.ts";

describe("channels", () => {
  test("safeChannelId rejects traversal", () => {
    expect(safeChannelId("../x")).toBeNull();
    expect(safeChannelId("good-id")).toBe("good-id");
  });

  test("post + drain + read", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "ap-ch-"));
    expect(postChannel(dataDir, "team", "agent:a", "hello")).toBe(true);
    const notes = drainChannelNotes();
    expect(notes.length).toBe(1);
    expect(notes[0]).toContain("hello");
    expect(drainChannelNotes().length).toBe(0);
    const rows = readChannel(dataDir, "team");
    expect(rows.length).toBe(1);
    expect(rows[0]!.from).toBe("agent:a");
  });
});
