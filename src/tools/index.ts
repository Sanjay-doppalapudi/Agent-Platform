// Tool registry. Order is FIXED (part of the stable prompt prefix for caching).
// Descriptions are one sentence — token budget.
import type { Config } from "../config.ts";
import type { ToolSchema } from "../provider.ts";
import { ToolError, truncateMiddle } from "./shared.ts";
import { readTool } from "./read.ts";
import { writeTool } from "./write.ts";
import { editTool } from "./edit.ts";
import { bashTool } from "./bash.ts";
import { globTool } from "./glob.ts";
import { grepTool } from "./grep.ts";

export interface ToolCtx {
  cwd: string;
  signal: AbortSignal;
  config: Config;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: object;
  readOnly: boolean;
  run(args: any, ctx: ToolCtx): Promise<string>;
}

const RESULT_BACKSTOP_BYTES = 40_000;

export const TOOLS: ToolDef[] = [
  {
    name: "read",
    description: "Read a text file with line numbers.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        offset: { type: "number", description: "1-based start line" },
        limit: { type: "number" },
      },
      required: ["path"],
    },
    readOnly: true,
    run: readTool,
  },
  {
    name: "write",
    description: "Write a file (creates parent dirs, overwrites).",
    parameters: {
      type: "object",
      properties: { path: { type: "string" }, content: { type: "string" } },
      required: ["path", "content"],
    },
    readOnly: false,
    run: writeTool,
  },
  {
    name: "edit",
    description: "Replace an exact string in a file; old must match uniquely unless all=true.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        old: { type: "string" },
        new: { type: "string" },
        all: { type: "boolean" },
      },
      required: ["path", "old", "new"],
    },
    readOnly: false,
    run: editTool,
  },
  {
    name: "bash",
    description: "Run a shell command; background=true for servers (returns pid+log).",
    parameters: {
      type: "object",
      properties: {
        cmd: { type: "string" },
        cwd: { type: "string" },
        timeout: { type: "number", description: "seconds, max 600" },
        background: { type: "boolean" },
      },
      required: ["cmd"],
    },
    readOnly: false,
    run: bashTool,
  },
  {
    name: "glob",
    description: "List files matching a glob pattern, newest first.",
    parameters: {
      type: "object",
      properties: { pattern: { type: "string" }, cwd: { type: "string" } },
      required: ["pattern"],
    },
    readOnly: true,
    run: globTool,
  },
  {
    name: "grep",
    description: "Regex search file contents (ripgrep); mode: content|files|count.",
    parameters: {
      type: "object",
      properties: {
        pattern: { type: "string" },
        path: { type: "string" },
        glob: { type: "string" },
        mode: { type: "string", enum: ["content", "files", "count"] },
        ignoreCase: { type: "boolean" },
        context: { type: "number" },
      },
      required: ["pattern"],
    },
    readOnly: true,
    run: grepTool,
  },
];

const byName = new Map(TOOLS.map((t) => [t.name, t]));

/** OpenAI tool schemas — built once, stable order/bytes. */
export const TOOL_SCHEMAS: ToolSchema[] = TOOLS.map((t) => ({
  type: "function",
  function: { name: t.name, description: t.description, parameters: t.parameters },
}));

// Plan mode: read-only subset, same fixed order — its own stable cache prefix.
const PLAN_SCHEMAS: ToolSchema[] = TOOLS.filter((t) => t.readOnly).map((t) => ({
  type: "function" as const,
  function: { name: t.name, description: t.description, parameters: t.parameters },
}));

export function toolSchemasFor(mode: "plan" | "code"): ToolSchema[] {
  return mode === "plan" ? PLAN_SCHEMAS : TOOL_SCHEMAS;
}

export function getTool(name: string): ToolDef | undefined {
  return byName.get(name);
}

/** Execute one tool call; returns model-facing result text (errors included). */
export async function execTool(
  name: string,
  rawArgs: string,
  ctx: ToolCtx,
): Promise<{ output: string; error: boolean }> {
  const tool = byName.get(name);
  if (!tool) return { output: `unknown tool: ${name}`, error: true };
  let args: any;
  try {
    args = rawArgs ? JSON.parse(rawArgs) : {};
  } catch {
    return { output: `invalid JSON arguments for ${name}`, error: true };
  }
  try {
    const output = await tool.run(args, ctx);
    return { output: truncateMiddle(output, RESULT_BACKSTOP_BYTES), error: false };
  } catch (e) {
    if (e instanceof ToolError) return { output: `error: ${e.message}`, error: true };
    return { output: `error: ${(e as Error).message}`, error: true };
  }
}
