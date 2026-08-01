// `ap tool <name> '<json>'` — direct tool tester.
import { loadConfig } from "./config.ts";
import { execTool } from "./tools/index.ts";
import type { CliFlags } from "./index.ts";

export async function toolMain(flags: CliFlags, rest: string[]) {
  const [name, json] = rest;
  if (!name) {
    console.error(`usage: ap tool <name> '<json-args>'`);
    process.exit(1);
  }
  const config = loadConfig(flags);
  const start = performance.now();
  const { output, error } = await execTool(name, json ?? "{}", {
    cwd: config.cwd,
    signal: new AbortController().signal,
    config,
  });
  const ms = Math.round(performance.now() - start);
  console.log(output);
  console.error(`[${name} ${error ? "error" : "ok"} in ${ms}ms]`);
  process.exit(error ? 1 : 0);
}
