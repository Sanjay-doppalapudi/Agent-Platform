// File-tool regressions from the tools audit. Two of these were silent and
// serious: .env redaction did nothing at all on Windows-style files, and edit
// could duplicate a file's entire tail while reporting success.
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { execTool } from "../src/tools/index.ts";
import { redactEnvContent } from "../src/tools/shared.ts";

function ws() {
  const cwd = mkdtempSync(join(tmpdir(), "ap-files-"));
  const config = loadConfig({ cwd } as any);
  return {
    cwd,
    ctx: { cwd, config, signal: new AbortController().signal, permit: async () => true } as any,
  };
}

describe("redactEnvContent line endings", () => {
  const SECRET = "sk-CANARY-must-not-appear";
  test("LF is masked", () => {
    expect(redactEnvContent(`KEY=${SECRET}\nB=2\n`)).not.toContain(SECRET);
  });
  // REGRESSION: splitting on "\n" left a trailing \r that the value pattern
  // could not match, so redaction silently no-opped on the primary platform.
  test("CRLF is masked", () => {
    expect(redactEnvContent(`KEY=${SECRET}\r\nB=2\r\n`)).not.toContain(SECRET);
  });
  test("CR-only is masked", () => {
    expect(redactEnvContent(`KEY=${SECRET}\rB=2\r`)).not.toContain(SECRET);
  });
  test("export-prefixed and spaced forms are masked", () => {
    expect(redactEnvContent(`export KEY = ${SECRET}\r\n`)).not.toContain(SECRET);
  });
  test("comments, blanks and empty values survive untouched", () => {
    const out = redactEnvContent("# a comment\r\n\r\nEMPTY=\r\nKEY=v\r\n");
    expect(out).toContain("# a comment");
    expect(out).toContain("EMPTY=");
    expect(out).toContain("KEY=***");
  });
});

describe("read on a CRLF .env", () => {
  test("secrets never reach the model", async () => {
    const w = ws();
    writeFileSync(join(w.cwd, ".env"), "OPENAI_API_KEY=sk-CRLF-CANARY\r\nDB_PASSWORD=hunter2\r\n");
    const r = await execTool("read", JSON.stringify({ path: ".env" }), w.ctx);
    expect(r.output).not.toContain("sk-CRLF-CANARY");
    expect(r.output).not.toContain("hunter2");
    expect(r.output).toContain("***");
  });
});

describe("edit does not interpret $ substitution patterns", () => {
  // REGRESSION: String.replace(old, string) expands $$, $&, $` and $'. `$'`
  // spliced the whole file tail in, silently duplicating it.
  const cases: [string, string][] = [
    ["dollar-tail ($')", "IFS=$'\\n'"],
    ["double dollar ($$)", 'echo "$$file"'],
    ["ampersand ($&)", "cost=$&100"],
    ["backtick ($`)", "x=$`y"],
  ];
  for (const [label, replacement] of cases) {
    test(`${label} is written literally`, async () => {
      const w = ws();
      const file = join(w.cwd, "s.sh");
      const tail = "rest of the file\nmore lines\n";
      writeFileSync(file, `#!/bin/sh\nMARKER\n${tail}`);
      const r = await execTool("edit", JSON.stringify({ path: "s.sh", old: "MARKER", new: replacement }), w.ctx);
      expect(r.error).toBe(false);
      const after = readFileSync(file, "utf8");
      expect(after).toBe(`#!/bin/sh\n${replacement}\n${tail}`);
      // the tail must appear exactly once — duplication was the symptom
      expect(after.split("more lines").length - 1).toBe(1);
    });
  }

  test("all:true path is also literal", async () => {
    const w = ws();
    const file = join(w.cwd, "m.txt");
    writeFileSync(file, "X\nX\n");
    await execTool("edit", JSON.stringify({ path: "m.txt", old: "X", new: "$$", all: true }), w.ctx);
    expect(readFileSync(file, "utf8")).toBe("$$\n$$\n");
  });
});

describe("read line numbering", () => {
  test("no phantom blank line past the end of a newline-terminated file", async () => {
    const w = ws();
    writeFileSync(join(w.cwd, "a.txt"), "one\ntwo\nthree\n");
    const r = await execTool("read", JSON.stringify({ path: "a.txt" }), w.ctx);
    expect(r.output).toContain("3: three");
    expect(r.output).not.toContain("4: ");
  });
  test("a file without a trailing newline reports every line", async () => {
    const w = ws();
    writeFileSync(join(w.cwd, "b.txt"), "one\ntwo");
    const r = await execTool("read", JSON.stringify({ path: "b.txt" }), w.ctx);
    expect(r.output).toContain("2: two");
    expect(r.output).not.toContain("3: ");
  });
});
