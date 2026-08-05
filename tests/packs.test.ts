// Extension packs: agent profiles, skills sources, MCP server spec merging.
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { agentsPromptBlock, discoverAgents, getAgent } from "../src/agents.ts";
import { loadConfig } from "../src/config.ts";
import { mcpServerSpecs } from "../src/mcp.ts";
import { parseSource } from "../src/skills.ts";

const tmp = mkdtempSync(join(tmpdir(), "ap-test-packs-"));
mkdirSync(join(tmp, ".ap", "agents"), { recursive: true });
writeFileSync(
  join(tmp, ".ap", "agents", "Reviewer.md"),
  `---\ndescription: read-only reviewer\nmodel: acme/fast-1\ntools: read, grep glob\n---\nReview code. Never write.\n`,
);
writeFileSync(join(tmp, ".ap", "agents", "empty.md"), `---\ndescription: no body\n---\n`);
const config = loadConfig({ cwd: tmp } as any);

describe("agent profiles", () => {
  test("frontmatter parsed, name lowercased from filename", () => {
    const a = getAgent(config, "reviewer")!;
    expect(a.description).toBe("read-only reviewer");
    expect(a.model).toBe("acme/fast-1");
    expect(a.tools).toEqual(["read", "grep", "glob"]); // comma AND space separated
    expect(a.body).toContain("Never write");
  });
  test("bodyless files are ignored", () => {
    expect(discoverAgents(config).map((a) => a.name)).toEqual(["reviewer"]);
  });
  test("prompt block advertises by name", () => {
    const block = agentsPromptBlock(discoverAgents(config));
    expect(block).toContain('"name": "reviewer"');
  });
  test("no defs → empty block (prompt byte-stability)", () => {
    expect(agentsPromptBlock([])).toBe("");
  });
});

describe("skills parseSource", () => {
  test("owner/repo", () => expect(parseSource("a/b")).toEqual({ owner: "a", repo: "b" }));
  test("github url", () => expect(parseSource("https://github.com/a/b.git").repo).toBe("b"));
  test("skills.sh url with skill", () => {
    expect(parseSource("https://www.skills.sh/a/b/c")).toEqual({ owner: "a", repo: "b", skill: "c" });
  });
  test("garbage throws with guidance", () => {
    expect(() => parseSource("not a source")).toThrow(/owner\/repo/);
  });
});

describe("mcpServerSpecs", () => {
  test("project .mcp.json merges over config.mcpServers and invalid entries drop", () => {
    writeFileSync(join(tmp, ".mcp.json"), JSON.stringify({
      mcpServers: {
        echo: { command: "bun", args: ["run", "x.ts"] },
        broken: { note: "no command or url" },
      },
    }));
    const cfg = loadConfig({ cwd: tmp } as any);
    cfg.mcpServers = { echo: { url: "https://overridden.example" }, extra: { url: "https://x" } };
    const specs = mcpServerSpecs(cfg);
    expect(specs["echo"]!.command).toBe("bun"); // .mcp.json wins
    expect(specs["extra"]!.url).toBe("https://x"); // config-only survives
    expect(specs["broken"]).toBeUndefined(); // invalid dropped
  });
});
