// `ap tool <name> '<json>'` — direct tool tester.
import { loadConfig } from "./config.ts";
import { execTool, getTool, resolveToolName } from "./tools/index.ts";
import type { CliFlags } from "./index.ts";

export async function toolMain(flags: CliFlags, rest: string[]) {
  const [name, json] = rest;
  if (!name) {
    console.error(`usage: ap tool <name> '<json-args>'`);
    process.exit(1);
  }
  const config = loadConfig(flags);
  if (!getTool(resolveToolName(name))) {
    // Not a built-in — it may live on an MCP server; connect and retry.
    const { initMcp } = await import("./mcp.ts");
    await initMcp(config, (m) => console.error(`⚠ ${m}`));
  }
  const start = performance.now();
  const { output, error } = await execTool(name, json ?? "{}", {
    cwd: config.cwd,
    signal: new AbortController().signal,
    config,
    permit: async () => true, // developer typed the exact args — trusted
    warn: (m) => console.error(`⚠ ${m}`),
  });
  const ms = Math.round(performance.now() - start);
  console.log(output);
  console.error(`[${name} ${error ? "error" : "ok"} in ${ms}ms]`);
  process.exit(error ? 1 : 0);
}
