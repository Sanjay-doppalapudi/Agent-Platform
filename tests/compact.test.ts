import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendArchive, listArchives, parseMemoryCards, restoreContextNote } from "../src/compact.ts";
import { Session } from "../src/session.ts";

describe("parseMemoryCards", () => {
  test("NONE yields empty", () => {
    expect(parseMemoryCards("NONE")).toEqual([]);
  });
  test("parses cards separated by ---", () => {
    const raw = `Title: Tabs
User wanted: use tabs
Why (guess): consistency
---
Title: Tests
User wanted: always add tests
Why (guess): regressions`;
    const cards = parseMemoryCards(raw);
    expect(cards.length).toBe(2);
    expect(cards[0]).toContain("Title: Tabs");
  });
});

describe("archives index", () => {
  test("append and list", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "ap-arch-"));
    appendArchive(dataDir, {
      at: "2026-01-01T00:00:00.000Z",
      oldId: "old1",
      newId: "new1",
      summaryChars: 42,
      reason: "manual",
      cwd: "/tmp/x",
    });
    const rows = listArchives(dataDir);
    expect(rows.length).toBe(1);
    expect(rows[0]!.oldId).toBe("old1");
    expect(rows[0]!.newId).toBe("new1");
  });
});

describe("restoreContextNote", () => {
  test("loads compacted summary from parent session", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "ap-restore-"));
    const s = Session.create(dataDir, {
      cwd: "/tmp",
      model: "x",
      at: new Date().toISOString(),
      parentSessionId: "prev",
    });
    s.append({ role: "user", content: "[Compacted context from session prev]\nGoals: ship it" });
    s.append({ role: "assistant", content: "Context loaded — continuing from the summary." });
    const note = restoreContextNote(dataDir, s.id);
    expect(note).toContain("Goals: ship it");
    expect(note).toContain(s.id);
  });
});
