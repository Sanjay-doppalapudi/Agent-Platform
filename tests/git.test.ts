// Real-git helpers for /commit: branch protection, slugs, message cleanup.
import { describe, expect, test } from "bun:test";
import { cleanCommitMessage, diffForPrompt, isProtectedBranch, parsePorcelain, slugifyBranch } from "../src/git.ts";

describe("parsePorcelain", () => {
  // Regression: a trimmed leading space must not eat the filename's first
  // char (" M a.txt" trimmed to "M a.txt" once sliced as ".txt").
  test("survives trimmed leading space", () => {
    expect(parsePorcelain("M a.txt\n?? b.txt")).toEqual(["a.txt", "b.txt"]);
  });
  test("standard two-column codes", () => {
    expect(parsePorcelain(" M src/x.ts\nA  new.ts\n?? untracked.md")).toEqual(["src/x.ts", "new.ts", "untracked.md"]);
  });
  test("renames report the destination", () => {
    expect(parsePorcelain("R  old.ts -> new.ts")).toEqual(["new.ts"]);
  });
  test("quoted paths are unwrapped, blanks skipped", () => {
    expect(parsePorcelain('?? "has space.txt"\n\n')).toEqual(["has space.txt"]);
  });
});

describe("isProtectedBranch", () => {
  for (const b of ["main", "master", "MAIN", "develop", "production", "release/1.2", "release-2024"]) {
    test(`protects ${b}`, () => expect(isProtectedBranch(b)).toBe(true));
  }
  for (const b of ["ap/fix-tests", "feature/x", "sanjay-dev", "releases"]) {
    test(`allows ${b}`, () => expect(isProtectedBranch(b)).toBe(false));
  }
});

describe("slugifyBranch", () => {
  test("builds a kebab slug under the ap/ prefix", () => {
    expect(slugifyBranch("Fix the failing date test")).toBe("ap/fix-the-failing-date-test");
  });
  test("caps word count and strips punctuation", () => {
    expect(slugifyBranch("add: input validation, to the signup flow please now!"))
      .toBe("ap/add-input-validation-to-the-signup");
  });
  test("never produces an empty or trailing-dash name", () => {
    expect(slugifyBranch("!!!")).toBe("ap/work");
    expect(slugifyBranch("")).toBe("ap/work");
    expect(slugifyBranch("x".repeat(80)).endsWith("-")).toBe(false);
  });
});

describe("cleanCommitMessage", () => {
  test("strips code fences, preamble, and quotes", () => {
    expect(cleanCommitMessage('```\nCommit message: "Add retry to fetch"\n```')).toBe("Add retry to fetch");
  });
  test("drops attribution/co-author lines", () => {
    const out = cleanCommitMessage("Fix parser\n\nWhy: handles CRLF\nCo-Authored-By: Someone <a@b.c>\n🤖 Generated with X");
    expect(out).toContain("Fix parser");
    expect(out).toContain("Why: handles CRLF");
    expect(out.toLowerCase()).not.toContain("co-authored-by");
    expect(out).not.toContain("🤖");
  });
  test("caps the subject at 72 chars and drops a trailing period", () => {
    const out = cleanCommitMessage("A".repeat(100) + ".");
    expect(out.split("\n")[0]!.length).toBeLessThanOrEqual(72);
  });
  test("keeps subject + body separated by a blank line", () => {
    expect(cleanCommitMessage("Subject here\n\nBody line")).toBe("Subject here\n\nBody line");
  });
  test("empty input stays empty (caller falls back)", () => expect(cleanCommitMessage("")).toBe(""));
});

describe("diffForPrompt", () => {
  test("short diffs pass through untouched", () => expect(diffForPrompt("a\nb")).toBe("a\nb"));
  test("long diffs are capped with a marker", () => {
    const out = diffForPrompt("x".repeat(20_000), 1000);
    expect(out.length).toBeLessThan(1100);
    expect(out).toContain("truncated");
  });
});
