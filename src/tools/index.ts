// Tool registry. Order is FIXED (part of the stable prompt prefix for caching).
// Descriptions are one sentence — token budget. Schemas must never change bytes.
import type { Config } from "../config.ts";
import type { ToolSchema } from "../provider.ts";
import { ToolError, truncateMiddle } from "./shared.ts";
import { readTool } from "./read.ts";
import { writeTool } from "./write.ts";
import { editTool } from "./edit.ts";
import { bashTool } from "./bash.ts";
import { globTool } from "./glob.ts";
import { grepTool } from "./grep.ts";

export interface PermitRequest {
  action: string;
  detail: string;
  path?: string;
}
export type PermitFn = (req: PermitRequest) => Promise<boolean>;

/** Default for headless contexts: deny — the ToolError text instructs the model. */
export const autoDenyPermit: PermitFn = async () => false;

export interface ToolCtx {
  cwd: string;
  signal: AbortSignal;
  config: Config;
  permit: PermitFn;
  warn?: (msg: string) => void;
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

// Weaker models invent tool names — map the common guesses to real tools.
const NAME_ALIASES: Record<string, string> = {
  search: "grep", rg: "grep",
  list: "glob", ls: "glob", find: "glob", find_files: "glob", list_files: "glob",
  shell: "bash", terminal: "bash", run_command: "bash", execute: "bash", exec: "bash", sh: "bash", cmd: "bash",
  create: "write", create_file: "write", write_file: "write", save: "write",
  str_replace: "edit", str_replace_editor: "edit", apply_edit: "edit", replace: "edit", edit_file: "edit",
  cat: "read", view: "read", read_file: "read", open: "read", open_file: "read",
};

/** Canonical tool name for a model-supplied name (exact wins; alias next). */
export function resolveToolName(name: string): string {
  if (byName.has(name)) return name;
  return NAME_ALIASES[name] ?? name;
}

// …and misname arguments. Map aliases onto canonical names (only when absent).
const ARG_ALIASES: Record<string, Record<string, string>> = {
  read: { file_path: "path", filePath: "path", filename: "path", file: "path", start: "offset", lines: "limit" },
  write: { file_path: "path", filePath: "path", filename: "path", file: "path", text: "content", contents: "content", body: "content", data: "content" },
  edit: { file_path: "path", filePath: "path", filename: "path", file: "path" },
  bash: { command: "cmd", script: "cmd", workdir: "cwd", working_dir: "cwd", workingDir: "cwd", directory: "cwd" },
  grep: { query: "pattern", regex: "pattern", q: "pattern", search: "pattern", ignore_case: "ignoreCase", case_insensitive: "ignoreCase", include: "glob", dir: "path" },
  glob: { globPattern: "pattern", glob: "pattern", path: "cwd", dir: "cwd", directory: "cwd" },
};

const NUMERIC_ARGS: Record<string, string[]> = {
  read: ["offset", "limit"],
  bash: ["timeout"],
  grep: ["context"],
};

function normalizeArgs(name: string, args: any): any {
  if (!args || typeof args !== "object") return {};
  const map = ARG_ALIASES[name];
  if (map) {
    for (const [alias, canon] of Object.entries(map)) {
      if (args[canon] === undefined && args[alias] !== undefined) args[canon] = args[alias];
    }
  }
  for (const k of NUMERIC_ARGS[name] ?? []) {
    const v = args[k];
    if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) args[k] = Number(v);
  }
  return args;
}

/** JSON.parse with cheap repairs for common model mistakes. Throws if hopeless. */
function parseArgsLenient(raw: string): any {
  if (!raw || !raw.trim()) return {};
  try { return JSON.parse(raw); } catch {}
  let s = raw.trim();
  try {
    const once = JSON.parse(s);
    if (typeof once === "string") return JSON.parse(once); // double-encoded
  } catch {}
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  s = s.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  s = s.replace(/,\s*([}\]])/g, "$1");
  return JSON.parse(s);
}

export function getTool(name: string): ToolDef | undefined {
  return byName.get(name);
}

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

/** Execute one tool call; returns model-facing result text (errors included). */
export async function execTool(
  name: string,
  rawArgs: string,
  ctx: ToolCtx,
): Promise<{ output: string; error: boolean }> {
  const canonical = resolveToolName(name);
  const tool = byName.get(canonical);
  if (!tool) {
    return { output: `unknown tool: ${name} — available tools: read, write, edit, bash, glob, grep`, error: true };
  }
  let args: any;
  try {
    args = parseArgsLenient(rawArgs);
  } catch {
    return {
      output: `invalid JSON arguments for ${canonical}: ${rawArgs.slice(0, 200)} — resend the call with valid JSON`,
      error: true,
    };
  }
  args = normalizeArgs(canonical, args);
  try {
    const output = await tool.run(args, ctx);
    return { output: truncateMiddle(output, RESULT_BACKSTOP_BYTES), error: false };
  } catch (e) {
    if (e instanceof ToolError) return { output: `error: ${e.message}`, error: true };
    return { output: `error: ${(e as Error).message}`, error: true };
  }
}
