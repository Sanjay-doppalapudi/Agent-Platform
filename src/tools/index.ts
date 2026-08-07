// Tool registry. Order is FIXED (part of the stable prompt prefix for caching).
// Descriptions are one sentence — token budget. Schemas must never change bytes.
//
// TRUST MODEL (see SECURITY.md): tool calls originate from the model inside a
// session the local user started and supervises. execTool normalizes names and
// arguments (aliases, lenient JSON repair) but never widens capability — it
// validates required arguments against each tool's schema below, and
// AUTHORIZATION is enforced inside the tools themselves: ensureAllowed/
// ensureReadable path checks, scanDangerous/scanCmdPaths for bash, hard-denied
// AP-private paths, and plan mode's structurally read-only schema sets.
import type { Config, PermissionVerdict } from "../config.ts";
import type { ToolSchema } from "../provider.ts";
import { ToolError, truncateMiddle } from "./shared.ts";
import { readTool } from "./read.ts";
import { writeTool } from "./write.ts";
import { editTool } from "./edit.ts";
import { bashTool } from "./bash.ts";
import { globTool } from "./glob.ts";
import { grepTool } from "./grep.ts";
import { agentTool } from "./agent.ts";
import { artifactTool } from "./artifact.ts";
import { fetchTool } from "./fetch.ts";
import { todoTool } from "./todo.ts";
import { websearchTool } from "./websearch.ts";

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
  /** Nested progress line (subagent activity) — rendered dim + indented. */
  subline?: (text: string) => void;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: object;
  readOnly: boolean;
  /** Safe to run concurrently with other tools (defaults to readOnly). */
  parallelSafe?: boolean;
  /** Excluded from the --light profile's schema set. */
  fullOnly?: boolean;
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
  {
    name: "agent",
    description: "Delegate an independent subtask to a parallel subagent; returns its final answer. background:true detaches it — the tool returns immediately and the result arrives with the next turn.",
    parameters: {
      type: "object",
      properties: {
        task: { type: "string", description: "complete, self-contained task description" },
        name: { type: "string", description: "named agent profile to run as (see Named agents)" },
        cwd: { type: "string" },
        timeout: { type: "number", description: "seconds, default 300" },
        background: { type: "boolean", description: "run detached; result is delivered as a note on the next turn" },
      },
      required: ["task"],
    },
    readOnly: false,
    parallelSafe: true, // children serialize their own mutations
    fullOnly: true,
    run: agentTool,
  },
  {
    name: "artifact",
    description: "Save a self-contained HTML page (report, diagram, dashboard) the user can open in a browser. Inline CSS/JS only — a no-network CSP is enforced.",
    parameters: {
      type: "object",
      properties: {
        title: { type: "string", description: "human title; also derives the filename" },
        html: { type: "string", description: "complete self-contained HTML (2MB cap, no external resources)" },
        slug: { type: "string", description: "optional filename slug [a-z0-9-]" },
      },
      required: ["title", "html"],
    },
    readOnly: false,
    parallelSafe: true, // each call writes a distinct timestamped file
    fullOnly: true,
    run: artifactTool,
  },
  {
    name: "fetch",
    description: "Fetch a URL and return its readable text (50KB cap); render=true for JS-heavy pages (headless system browser).",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string" },
        render: { type: "boolean", description: "render JavaScript via installed Chrome/Edge" },
      },
      required: ["url"],
    },
    readOnly: true,
    fullOnly: true,
    run: fetchTool,
  },
  {
    name: "todo",
    description: "Track a task checklist: add items, mark done by number, clear.",
    parameters: {
      type: "object",
      properties: {
        add: { type: "array", items: { type: "string" } },
        done: { type: "array", items: { type: "number" } },
        clear: { type: "boolean" },
      },
    },
    readOnly: true,
    fullOnly: true,
    run: todoTool,
  },
  {
    name: "websearch",
    description: "Search the web (DuckDuckGo): titles, URLs, snippets — then fetch the promising ones.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string" },
        limit: { type: "number", description: "max results, default 8" },
      },
      required: ["query"],
    },
    readOnly: true,
    fullOnly: true,
    run: websearchTool,
  },
];

const byName = new Map(TOOLS.map((t) => [t.name, t]));

// Dynamic tools (MCP servers) — registered ONCE per process by mcp.ts before
// the first turn, then frozen: their schemas append after the built-ins in a
// fixed order, so the request's tool list stays byte-stable for the whole
// session and provider prefix caching still hits. Never present in --light.
const dynamicTools = new Map<string, ToolDef>();
const dynamicAliases = new Map<string, string>();
let dynamicSchemasCode: ToolSchema[] = [];
let dynamicSchemasPlan: ToolSchema[] = [];

export function registerDynamicTools(defs: ToolDef[], aliases: Record<string, string>) {
  for (const d of defs) dynamicTools.set(d.name, d);
  for (const [a, c] of Object.entries(aliases)) {
    if (!byName.has(a) && !dynamicAliases.has(a)) dynamicAliases.set(a, c);
  }
  const all = [...dynamicTools.values()];
  dynamicSchemasCode = all.map(toSchema);
  dynamicSchemasPlan = all.filter((t) => t.readOnly).map(toSchema);
}

// Weaker models invent tool names — map the common guesses to real tools.
const NAME_ALIASES: Record<string, string> = {
  search: "grep", rg: "grep",
  list: "glob", ls: "glob", find: "glob", find_files: "glob", list_files: "glob",
  shell: "bash", terminal: "bash", run_command: "bash", execute: "bash", exec: "bash", sh: "bash", cmd: "bash",
  create: "write", create_file: "write", write_file: "write", save: "write",
  str_replace: "edit", str_replace_editor: "edit", apply_edit: "edit", replace: "edit", edit_file: "edit",
  cat: "read", view: "read", read_file: "read", open: "read", open_file: "read",
  web_search: "websearch", search_web: "websearch", google: "websearch", web: "websearch", internet_search: "websearch", duckduckgo: "websearch",
  browse: "fetch", browser: "fetch", visit: "fetch", get_url: "fetch",
};

/** Canonical tool name for a model-supplied name (exact wins; alias next). */
export function resolveToolName(name: string): string {
  if (byName.has(name)) return name;
  if (NAME_ALIASES[name]) return NAME_ALIASES[name];
  if (dynamicTools.has(name)) return name;
  return dynamicAliases.get(name) ?? name;
}

// …and misname arguments. Map aliases onto canonical names (only when absent).
const ARG_ALIASES: Record<string, Record<string, string>> = {
  read: { file_path: "path", filePath: "path", filename: "path", file: "path", start: "offset", lines: "limit" },
  write: { file_path: "path", filePath: "path", filename: "path", file: "path", text: "content", contents: "content", body: "content", data: "content" },
  edit: { file_path: "path", filePath: "path", filename: "path", file: "path" },
  bash: { command: "cmd", script: "cmd", workdir: "cwd", working_dir: "cwd", workingDir: "cwd", directory: "cwd" },
  grep: { query: "pattern", regex: "pattern", q: "pattern", search: "pattern", ignore_case: "ignoreCase", case_insensitive: "ignoreCase", include: "glob", dir: "path" },
  glob: { globPattern: "pattern", glob: "pattern", path: "cwd", dir: "cwd", directory: "cwd" },
  agent: { prompt: "task", description: "task", instructions: "task", subtask: "task" },
  fetch: { link: "url", uri: "url", href: "url", javascript: "render", js: "render" },
  websearch: { q: "query", search: "query", term: "query", keywords: "query", text: "query", max_results: "limit", maxResults: "limit", n: "limit", count: "limit" },
};

const NUMERIC_ARGS: Record<string, string[]> = {
  read: ["offset", "limit"],
  bash: ["timeout"],
  grep: ["context"],
  websearch: ["limit"],
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
export function parseArgsLenient(raw: string): any {
  if (!raw || !raw.trim()) return {};
  let s = raw.trim();
  try {
    const once = JSON.parse(s);
    if (typeof once !== "string") return once;
    // Valid JSON that IS a string = double-encoded arguments. (This used to be
    // unreachable: an earlier `return JSON.parse(raw)` handed the raw string
    // back to callers, which then saw no arguments at all.)
    s = once;
    try { return JSON.parse(s); } catch {} // fall through to the repairs below
  } catch {}
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  s = s.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  s = s.replace(/,\s*([}\]])/g, "$1");
  return JSON.parse(s);
}

export function getTool(name: string): ToolDef | undefined {
  return byName.get(name) ?? dynamicTools.get(name);
}

const toSchema = (t: ToolDef): ToolSchema => ({
  type: "function",
  function: { name: t.name, description: t.description, parameters: t.parameters },
});

// Four fixed schema sets (full/light × code/plan) — built once, stable
// order/bytes per profile, which is what makes prefix caching hit.
const FULL_CODE: ToolSchema[] = TOOLS.map(toSchema);
const FULL_PLAN: ToolSchema[] = TOOLS.filter((t) => t.readOnly).map(toSchema);
const LIGHT_CODE: ToolSchema[] = TOOLS.filter((t) => !t.fullOnly).map(toSchema);
const LIGHT_PLAN: ToolSchema[] = TOOLS.filter((t) => !t.fullOnly && t.readOnly).map(toSchema);

export const TOOL_SCHEMAS = FULL_CODE; // back-compat export

export function toolSchemasFor(config: Config): ToolSchema[] {
  let set: ToolSchema[];
  if (config.light) set = config.mode === "plan" ? LIGHT_PLAN : LIGHT_CODE;
  else if (config.mode === "plan") {
    set = dynamicSchemasPlan.length ? [...FULL_PLAN, ...dynamicSchemasPlan] : FULL_PLAN;
  } else {
    set = dynamicSchemasCode.length ? [...FULL_CODE, ...dynamicSchemasCode] : FULL_CODE;
  }
  // Named-agent tool whitelist (constant per process → bytes stay stable).
  if (config.toolFilter?.length) {
    const allow = new Set(config.toolFilter.map(resolveToolName));
    const filtered = set.filter((s) => allow.has(s.function.name));
    if (filtered.length) set = filtered;
  }
  return set;
}

/** Safe to run concurrently (read-only tools + explicitly parallel-safe ones). */
export function isParallelSafe(name: string): boolean {
  const t = getTool(resolveToolName(name));
  return !!t && (t.parallelSafe ?? t.readOnly);
}

// ---- Per-tool permission rules (config.permission) ------------------------

const globCache = new Map<string, RegExp>();
function globRe(pat: string): RegExp {
  let re = globCache.get(pat);
  if (!re) {
    const esc = pat.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
    re = new RegExp(`^${esc}$`, "i");
    globCache.set(pat, re);
  }
  return re;
}

/**
 * Evaluate config.permission for a tool call: "allow" | "ask" | "deny", or
 * null when no rule matches (caller falls back to the permissions mode).
 * Tool keys match exactly or by * glob (first matching key wins, definition
 * order). A record value holds command patterns matched against bash's cmd
 * (full-string globs — write trailing *), with "*" as the default.
 */
export function permissionFor(config: Config, tool: string, argsRaw: string): PermissionVerdict | null {
  const rules = config.permission;
  if (!rules) return null;
  for (const [key, val] of Object.entries(rules)) {
    if (key !== tool && !globRe(key).test(tool)) continue;
    if (typeof val === "string") return val;
    // Command patterns must be matched against the SAME arguments execTool
    // will run. Using strict JSON.parse here was a bypass: malformed args
    // (trailing comma, fences, smart quotes — which models emit constantly)
    // failed to parse, read as an empty command, missed every deny pattern,
    // and fell through to the default — after which the lenient parser in
    // execTool repaired them and ran the command anyway.
    let cmd: string | null = null;
    try {
      const a = parseArgsLenient(argsRaw);
      cmd = String(a?.cmd ?? a?.command ?? a?.script ?? "").trim();
    } catch {
      cmd = null; // unparseable even leniently
    }
    if (cmd === null) {
      // The command is unknowable, so the patterns cannot be evaluated. Fail
      // closed whenever any rule is stricter than the default.
      const strict = Object.entries(val).some(([p, v]) => p !== "*" && (v === "deny" || v === "ask"));
      return strict ? "ask" : val["*"] ?? null;
    }
    for (const [pat, v] of Object.entries(val)) {
      if (pat !== "*" && globRe(pat).test(cmd)) return v;
    }
    if (val["*"]) return val["*"];
    return null;
  }
  return null;
}

/** Execute one tool call; returns model-facing result text (errors included). */
export async function execTool(
  name: string,
  rawArgs: string,
  ctx: ToolCtx,
): Promise<{ output: string; error: boolean }> {
  const canonical = resolveToolName(name);
  const tool = getTool(canonical);
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
  // Schema validation: every required argument must be present after alias
  // normalization — a malformed call fails fast here, never inside a tool.
  const required: string[] = (tool.parameters as any)?.required ?? [];
  const missing = required.filter((k) => args[k] === undefined || args[k] === null);
  if (missing.length) {
    return {
      output: `missing required argument${missing.length > 1 ? "s" : ""} for ${canonical}: ${missing.join(", ")} — resend the call with all required fields`,
      error: true,
    };
  }
  try {
    const output = await tool.run(args, ctx);
    return { output: truncateMiddle(output, RESULT_BACKSTOP_BYTES), error: false };
  } catch (e) {
    if (e instanceof ToolError) return { output: `error: ${e.message}`, error: true };
    return { output: `error: ${(e as Error).message}`, error: true };
  }
}
