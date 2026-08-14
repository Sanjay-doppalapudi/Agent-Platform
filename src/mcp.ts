// MCP (Model Context Protocol) client — zero deps. JSON-RPC 2.0 over stdio
// (Bun.spawn, newline-delimited) or Streamable HTTP (plain fetch). Server
// configs come from `mcpServers` in ap.config.json / <dataDir>/config.json
// or a project .mcp.json (Claude Code's exact format — paste and go), so the
// existing MCP ecosystem works unchanged.
//
// Tools from connected servers register as DYNAMIC tools (tools/index.ts)
// named mcp_<server>_<tool>, appended after the built-ins in a fixed sorted
// order and frozen for the process lifetime — the schema list stays
// byte-stable, so provider prefix caching still hits. Connection happens
// once, lazily, before the first turn — never on the startup path. A dead
// server degrades to a one-line warning, never a crash. Full profile only.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Config, McpServerSpec } from "./config.ts";
import { clearDynamicTools, registerDynamicTools, type ToolDef } from "./tools/index.ts";
import { ToolError, truncateMiddle } from "./tools/shared.ts";
import { auditTask, finishTask, registerTask } from "./tasks.ts";

interface JsonRpcMsg {
  jsonrpc: "2.0";
  id?: number;
  method?: string;
  params?: any;
  result?: any;
  error?: { code: number; message: string };
}

export interface McpToolInfo { name: string; canonical: string; description: string; readOnly: boolean }
export interface McpServerStatus {
  name: string;
  transport: "stdio" | "http";
  ok: boolean;
  error?: string;
  /** Server-reported name + version from the initialize handshake. */
  serverName?: string;
  tools: McpToolInfo[];
}

const PROTOCOL_VERSION = "2025-06-18";
const CONNECT_TIMEOUT_MS = 30_000; // npx-style servers download on first spawn
const LIST_TIMEOUT_MS = 20_000;
const CALL_TIMEOUT_MS = 120_000;
const RESULT_CAP_BYTES = 40_000;
const MAX_TOOLS_PER_SERVER = 500;
/** Hard cap on a single stdio/SSE/HTTP JSON-RPC frame from an MCP server. */
const MAX_MCP_FRAME_BYTES = 2 * 1024 * 1024;

function anySignal(a: AbortSignal | undefined, b: AbortSignal): AbortSignal {
  if (!a) return b;
  const any = (AbortSignal as any).any;
  return typeof any === "function" ? any([a, b]) : b;
}

class McpClient {
  readonly transport: "stdio" | "http";
  serverName?: string;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private proc: any = null;
  private stderrTail = "";
  private httpSessionId: string | null = null;
  private negotiated = "";

  constructor(public name: string, private spec: McpServerSpec, private cwd: string) {
    this.transport = spec.command ? "stdio" : "http";
  }

  async start(): Promise<void> {
    if (this.transport === "stdio") this.spawn();
    const init = await this.request("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "ap", version: "1.0" },
    }, CONNECT_TIMEOUT_MS);
    this.negotiated = init?.protocolVersion ?? PROTOCOL_VERSION;
    if (init?.serverInfo?.name) {
      this.serverName = `${init.serverInfo.name}${init.serverInfo.version ? ` ${init.serverInfo.version}` : ""}`;
    }
    await this.notify("notifications/initialized");
  }

  private spawn() {
    const cmd = this.spec.command!;
    let argv = [cmd, ...(this.spec.args ?? [])];
    // Windows: npx and friends are .cmd shims — they only run under cmd /c.
    if (process.platform === "win32") {
      const found = Bun.which(cmd);
      if (found && /\.(cmd|bat)$/i.test(found)) argv = ["cmd", "/c", ...argv];
    }
    this.proc = Bun.spawn(argv, {
      cwd: this.cwd,
      env: { ...process.env, ...this.spec.env },
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
      windowsHide: true,
    } as any);
    this.readLoop();
    this.drainStderr();
  }

  private async readLoop() {
    const decoder = new TextDecoder();
    let buf = "";
    try {
      for await (const chunk of this.proc.stdout as any) {
        buf += decoder.decode(chunk, { stream: true });
        if (buf.length > MAX_MCP_FRAME_BYTES) {
          this.close();
          throw new Error(`mcp server "${this.name}" sent an oversized frame (>${MAX_MCP_FRAME_BYTES} bytes)`);
        }
        let nl: number;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          if (line.length > MAX_MCP_FRAME_BYTES) continue;
          try { this.dispatch(JSON.parse(line)); } catch {}
        }
      }
    } catch {}
    const err = new Error(
      `mcp server "${this.name}" exited${this.stderrTail ? ` — stderr: ${this.stderrTail.trim().slice(-300)}` : ""}`,
    );
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  private async drainStderr() {
    const decoder = new TextDecoder();
    try {
      for await (const chunk of this.proc.stderr as any) {
        this.stderrTail = (this.stderrTail + decoder.decode(chunk)).slice(-2000);
      }
    } catch {}
  }

  private dispatch(msg: JsonRpcMsg) {
    if (msg.id != null && msg.method === undefined) {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error.message ?? `RPC error ${msg.error.code}`));
      else p.resolve(msg.result);
      return;
    }
    if (msg.method && msg.id != null) {
      // Server-to-client request: we implement none — answer ping, refuse the rest.
      this.write(msg.method === "ping"
        ? { jsonrpc: "2.0", id: msg.id, result: {} }
        : { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "not supported by ap" } });
    }
    // Notifications (logging, progress) are ignored.
  }

  private write(msg: object) {
    try {
      this.proc.stdin.write(JSON.stringify(msg) + "\n");
      this.proc.stdin.flush();
    } catch {}
  }

  private request(method: string, params: any, timeoutMs: number, signal?: AbortSignal): Promise<any> {
    if (this.transport === "http") {
      return this.httpRoundtrip({ jsonrpc: "2.0", id: this.nextId++, method, params }, timeoutMs, signal);
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(
          `mcp "${this.name}" timed out on ${method} after ${timeoutMs / 1000}s` +
          (this.stderrTail ? ` — stderr: ${this.stderrTail.trim().slice(-300)}` : ""),
        ));
      }, timeoutMs);
      const onAbort = () => { this.pending.delete(id); clearTimeout(timer); reject(new Error("aborted")); };
      signal?.addEventListener("abort", onAbort, { once: true });
      const settle = <T,>(fn: (v: T) => void) => (v: T) => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        fn(v);
      };
      this.pending.set(id, { resolve: settle(resolve), reject: settle(reject) });
      this.write({ jsonrpc: "2.0", id, method, params });
    });
  }

  private async notify(method: string, params?: any): Promise<void> {
    const msg: JsonRpcMsg = { jsonrpc: "2.0", method, ...(params ? { params } : {}) };
    if (this.transport === "http") {
      await this.httpPost(msg, 10_000).then((r) => r.body?.cancel()).catch(() => {});
      return;
    }
    this.write(msg);
  }

  // --- Streamable HTTP -----------------------------------------------------
  private async httpPost(msg: object, timeoutMs: number, signal?: AbortSignal): Promise<Response> {
    const res = await fetch(this.spec.url!, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        ...(this.negotiated ? { "mcp-protocol-version": this.negotiated } : {}),
        ...(this.httpSessionId ? { "mcp-session-id": this.httpSessionId } : {}),
        ...this.spec.headers,
      },
      body: JSON.stringify(msg),
      signal: anySignal(signal, AbortSignal.timeout(timeoutMs)),
    });
    const sid = res.headers.get("mcp-session-id");
    if (sid) this.httpSessionId = sid;
    return res;
  }

  private async httpRoundtrip(msg: JsonRpcMsg, timeoutMs: number, signal?: AbortSignal): Promise<any> {
    const res = await this.httpPost(msg, timeoutMs, signal);
    if (!res.ok) {
      res.body?.cancel();
      throw new Error(`mcp "${this.name}" HTTP ${res.status} from ${this.spec.url}`);
    }
    const ctype = res.headers.get("content-type") ?? "";
    const reply = ctype.includes("text/event-stream")
      ? await readSseResponse(res, msg.id!)
      : await readJsonCapped(res, MAX_MCP_FRAME_BYTES, this.name);
    if (!reply) throw new Error(`mcp "${this.name}" stream ended without answering ${msg.method}`);
    if (reply.error) throw new Error(reply.error.message ?? `RPC error ${reply.error.code}`);
    return reply.result;
  }

  // --- MCP methods ---------------------------------------------------------
  async listTools(): Promise<any[]> {
    const tools: any[] = [];
    let cursor: string | undefined;
    // A server that returns a cursor forever (repeated, or fresh-but-empty
    // pages) used to spin here with no timeout and no exit — freezing the
    // REPL before its first turn, ACP session/new, and `ap serve` before the
    // port opened. Three independent bounds, all above the tool budget so a
    // legitimately fine-grained paginator is never truncated.
    const seen = new Set<string>();
    let pages = 0;
    do {
      const r = await this.request("tools/list", cursor ? { cursor } : {}, LIST_TIMEOUT_MS);
      const batch = r?.tools ?? [];
      tools.push(...batch);
      const next = r?.nextCursor;
      if (!next) break;
      if (typeof next !== "string" || seen.has(next)) break; // repeated/invalid cursor
      if (++pages > MAX_TOOLS_PER_SERVER + 50) break;        // runaway paginator
      if (!batch.length && pages > 20) break;                // empty pages forever
      seen.add(next);
      cursor = next;
    } while (tools.length < MAX_TOOLS_PER_SERVER);
    return tools;
  }

  async callTool(tool: string, args: any, signal: AbortSignal): Promise<string> {
    const r = await this.request("tools/call", { name: tool, arguments: args ?? {} }, CALL_TIMEOUT_MS, signal);
    const parts: string[] = [];
    for (const c of r?.content ?? []) {
      if (c?.type === "text") parts.push(c.text ?? "");
      else if (c?.type === "resource") parts.push(c.resource?.text ?? `[resource: ${c.resource?.uri ?? "?"}]`);
      else if (c?.type === "resource_link") parts.push(`[resource: ${c.uri ?? "?"}]`);
      else parts.push(`[${c?.type ?? "?"} content omitted]`);
    }
    let text = parts.join("\n").trim();
    if (!text && r?.structuredContent !== undefined) text = JSON.stringify(r.structuredContent);
    if (r?.isError) throw new ToolError(text || `mcp tool ${tool} reported an error`);
    return text || "(empty result)";
  }

  close() {
    try { this.proc?.kill(); } catch {}
  }
}

/** Read SSE events off an HTTP response until the JSON-RPC reply with `id`. */
async function readSseResponse(res: Response, id: number): Promise<JsonRpcMsg | null> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return null;
      buf = (buf + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      if (buf.length > MAX_MCP_FRAME_BYTES) {
        throw new Error(`mcp SSE frame exceeded ${MAX_MCP_FRAME_BYTES} bytes`);
      }
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const event = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const data = event
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim())
          .join("\n");
        if (!data) continue;
        try {
          const msg = JSON.parse(data) as JsonRpcMsg;
          if (msg.id === id) return msg;
        } catch {}
      }
    }
  } finally {
    try { reader.cancel(); } catch {}
  }
}

/** Bounded JSON body read — never buffer an unbounded MCP HTTP response. */
async function readJsonCapped(res: Response, limit: number, name: string): Promise<JsonRpcMsg> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error(`mcp "${name}" empty HTTP body`);
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > limit) {
        try { await reader.cancel(); } catch {}
        throw new Error(`mcp "${name}" HTTP body exceeded ${limit} bytes`);
      }
      chunks.push(value);
    }
  } finally {
    try { reader.releaseLock(); } catch {}
  }
  const buf = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.byteLength; }
  return JSON.parse(new TextDecoder().decode(buf)) as JsonRpcMsg;
}

// ---------------------------------------------------------------------------
// Discovery + registration
// ---------------------------------------------------------------------------

function readJson(path: string): any {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

/** Walk upward from cwd for .mcp.json (Claude Code's project MCP file). */
function findMcpJson(startDir: string): any {
  let dir = resolve(startDir);
  for (let i = 0; i < 20; i++) {
    const p = join(dir, ".mcp.json");
    if (existsSync(p)) return readJson(p);
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

/** All configured servers: home/ap config `mcpServers` + project `.mcp.json`
 *  when the workspace is trusted. Untrusted `.mcp.json` is ignored — that
 *  file is RCE on first turn (stdio spawn). */
export function mcpServerSpecs(config: Config): Record<string, McpServerSpec> {
  const merged: Record<string, McpServerSpec> = {
    ...(config.mcpServers ?? {}),
  };
  if (config.workspaceTrusted === true) {
    Object.assign(merged, findMcpJson(config.cwd)?.mcpServers ?? {});
  }
  const out: Record<string, McpServerSpec> = {};
  for (const [name, spec] of Object.entries(merged)) {
    if (spec && typeof spec === "object" && (spec.command || spec.url)) out[name] = spec;
  }
  return out;
}

const statuses: McpServerStatus[] = [];
const clients: McpClient[] = [];
let initPromise: Promise<void> | null = null;
let exitHookInstalled = false;

/** Connection results after initMcp resolves (for /mcp and `ap mcp`). */
export function mcpStatus(): McpServerStatus[] {
  return statuses;
}

const san = (s: string) => s.replace(/[^A-Za-z0-9_-]/g, "_");

/**
 * Connect every configured server and register its tools as dynamic tools.
 * Idempotent per process; front-ends await it before the FIRST turn so the
 * schema list is complete, then frozen for the session (cache stability).
 * A server that fails to connect is warned about and skipped — never fatal.
 */
export function initMcp(config: Config, warn: (msg: string) => void): Promise<void> {
  if (config.light) return Promise.resolve();
  if (initPromise) return initPromise;
  initPromise = doInit(config, warn).catch((e) => {
    warn(`mcp: init failed — ${(e as Error).message}`);
  });
  return initPromise;
}

async function doInit(config: Config, warn: (msg: string) => void): Promise<void> {
  const specs = mcpServerSpecs(config);
  const names = Object.keys(specs).sort(); // fixed order → stable schema bytes
  if (!names.length) return;

  const connected = new Map<string, { client: McpClient; tools: any[] }>();
  await Promise.all(names.map(async (name) => {
    const client = new McpClient(name, specs[name]!, config.cwd);
    try {
      await client.start();
      const tools = await client.listTools();
      clients.push(client);
      connected.set(name, { client, tools });
    } catch (e) {
      client.close();
      statuses.push({ name, transport: client.transport, ok: false, error: (e as Error).message, tools: [] });
      warn(`mcp: server "${name}" unavailable — ${(e as Error).message}`);
    }
  }));

  if (clients.length && !exitHookInstalled) {
    exitHookInstalled = true;
    process.on("exit", () => { for (const c of clients) c.close(); });
  }

  const defs: ToolDef[] = [];
  const aliases: Record<string, string> = {};
  const taken = new Set<string>();
  for (const name of names) {
    const r = connected.get(name);
    if (!r) continue;
    const status: McpServerStatus = { name, transport: r.client.transport, ok: true, serverName: r.client.serverName, tools: [] };
    for (const t of r.tools) {
      if (!t?.name) continue;
      // Provider function-name rules: [A-Za-z0-9_-], max 64 chars, unique.
      let canonical = `mcp_${san(name)}_${san(t.name)}`.slice(0, 64);
      for (let n = 2; taken.has(canonical); n++) canonical = `${canonical.slice(0, 61)}_${n}`;
      taken.add(canonical);
      const readOnly = false; // MCP annotations are untrusted hints — never
      // authorize plan-mode / parallel-safety from readOnlyHint alone.
      // A local allowlist can be added later; default fail-closed.
      const client = r.client;
      const toolName = t.name as string;
      defs.push({
        name: canonical,
        description: `[${name}] ${String(t.description ?? "").replace(/\s+/g, " ").trim()}`.slice(0, 300),
        parameters: t.inputSchema && typeof t.inputSchema === "object" ? t.inputSchema : { type: "object", properties: {} },
        readOnly,
        fullOnly: true,
        run: (args, ctx) => runMcpTool(client, toolName, args, ctx, config, name),
      });
      // Models say server.tool / server__tool / bare tool — accept all of them
      // (built-in names and aliases still win at resolution time).
      for (const a of [`${name}.${toolName}`, `${name}/${toolName}`, `${name}__${toolName}`, `mcp__${name}__${toolName}`, toolName]) {
        if (aliases[a] === undefined) aliases[a] = canonical;
        else if (aliases[a] !== canonical) aliases[a] = ""; // ambiguous — drop below
      }
      status.tools.push({ name: toolName, canonical, description: String(t.description ?? ""), readOnly });
    }
    statuses.push(status);
  }
  for (const k of Object.keys(aliases)) if (!aliases[k]) delete aliases[k];
  if (defs.length) registerDynamicTools(defs, aliases);
}

async function runMcpTool(
  client: McpClient,
  toolName: string,
  args: any,
  ctx: { signal: AbortSignal; subline?: (t: string) => void },
  config: Config,
  serverName: string,
): Promise<string> {
  const softMs = config.mcpAutoBackgroundMs ?? 30_000;
  const ctrl = new AbortController();
  const onTurnAbort = () => ctrl.abort();
  ctx.signal.addEventListener("abort", onTurnAbort, { once: true });
  const hardTimer = setTimeout(() => ctrl.abort(), CALL_TIMEOUT_MS);
  const work = client.callTool(toolName, args, ctrl.signal).finally(() => {
    clearTimeout(hardTimer);
    ctx.signal.removeEventListener("abort", onTurnAbort);
  });

  if (softMs <= 0) {
    return truncateMiddle(await work, RESULT_CAP_BYTES);
  }

  const winner = await Promise.race([
    work.then((r) => ({ t: "ok" as const, r })),
    new Promise<{ t: "soft" }>((res) => setTimeout(() => res({ t: "soft" }), softMs)),
  ]);
  if (winner.t === "ok") return truncateMiddle(winner.r, RESULT_CAP_BYTES);

  // Soft timeout: keep the in-flight call, detach from turn abort, report via tasks.
  ctx.signal.removeEventListener("abort", onTurnAbort);
  const label = `[mcp ${serverName}] ${toolName}`;
  const t = registerTask(label, CALL_TIMEOUT_MS);
  ctx.subline?.(`◇ MCP ${toolName} backgrounded as task #${t.id}`);
  auditTask(config, t, "start");
  void work.then((r) => {
    finishTask(t, "done", truncateMiddle(r, 20_000));
    auditTask(config, t, "end");
  }).catch((e) => {
    const msg = String((e as Error).message ?? e);
    finishTask(t, ctrl.signal.aborted ? "killed" : "error", msg.slice(0, 2000));
    auditTask(config, t, "end");
  });
  return `backgrounded MCP tool ${toolName} as task #${t.id} — the result will arrive with the next turn (/tasks to inspect)`;
}

/** Reconnect MCP servers and rebuild dynamic tool schemas (cache miss). */
export async function reloadMcp(config: Config, warn: (msg: string) => void): Promise<void> {
  if (config.light) {
    warn("mcp reload skipped in --light profile");
    return;
  }
  for (const c of clients) {
    try { c.close(); } catch {}
  }
  clients.length = 0;
  statuses.length = 0;
  clearDynamicTools();
  initPromise = null;
  await initMcp(config, warn);
}
