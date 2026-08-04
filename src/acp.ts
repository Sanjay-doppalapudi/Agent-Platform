// ACP (Agent Client Protocol) adapter — `ap acp`. Zed (or any ACP editor)
// spawns this process and drives it over stdio: newline-delimited JSON-RPC
// 2.0, protocol version 1 (https://agentclientprotocol.com). This is just
// another renderer of the AgentEvent stream: text → agent_message_chunk,
// reasoning → agent_thought_chunk, tool_start/end → tool_call updates, and
// sandbox permits → session/request_permission (native editor dialogs).
// stdout is PROTOCOL-ONLY — every log line goes to stderr.
//
// Zed settings.json:
//   { "agent_servers": { "AP": { "command": "ap", "args": ["acp"] } } }
import { loadConfig, resolveProvider, type Config, type McpServerSpec, type ResolvedProvider } from "./config.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { Checkpoints } from "./checkpoint.ts";
import { initMcp } from "./mcp.ts";
import { Session } from "./session.ts";
import { getTool } from "./tools/index.ts";
import { toolLabel } from "./ui.ts";
import type { CliFlags } from "./index.ts";

const ACP_VERSION = 1;

interface AcpLive {
  session: Session;
  config: Config;
  provider: ResolvedProvider;
  cp: Checkpoints;
  ctrl: AbortController | null;
  chain: Promise<unknown>; // one prompt at a time per session
}

// AgentEvent tool name → ACP ToolKind (drives the editor's icon).
const TOOL_KIND: Record<string, string> = {
  read: "read", grep: "search", glob: "search",
  edit: "edit", write: "edit",
  bash: "execute",
  fetch: "fetch", websearch: "fetch",
  agent: "other", todo: "think",
};

/** ACP prompt content blocks → one user-message string for the agent loop. */
function flattenPrompt(blocks: any[]): string {
  const parts: string[] = [];
  for (const b of blocks ?? []) {
    if (b?.type === "text") parts.push(b.text ?? "");
    else if (b?.type === "resource" && b.resource?.text != null) {
      parts.push(`<context uri="${b.resource.uri ?? ""}">\n${b.resource.text}\n</context>`);
    } else if (b?.type === "resource_link") parts.push(`@${b.uri ?? b.name ?? ""}`);
    // image/audio: declined via promptCapabilities
  }
  return parts.join("\n").trim();
}

/** ACP mcpServers entries → our McpServerSpec map (env/headers arrive as
 *  [{name, value}] pairs; tolerate plain records too). */
function acpMcpSpecs(list: any[]): Record<string, McpServerSpec> {
  const out: Record<string, McpServerSpec> = {};
  const pairs = (v: any): Record<string, string> | undefined => {
    if (!v) return undefined;
    if (Array.isArray(v)) {
      const rec: Record<string, string> = {};
      for (const p of v) if (p?.name) rec[p.name] = String(p.value ?? "");
      return rec;
    }
    return typeof v === "object" ? v : undefined;
  };
  for (const s of list ?? []) {
    if (!s?.name) continue;
    if (s.command) out[s.name] = { command: s.command, args: s.args ?? [], env: pairs(s.env) };
    else if (s.url) out[s.name] = { url: s.url, headers: pairs(s.headers) };
  }
  return out;
}

export async function acpMain(flags: CliFlags) {
  const sessions = new Map<string, AcpLive>();
  const log = (m: string) => process.stderr.write(`[acp] ${m}\n`);

  const out = (msg: object) => process.stdout.write(JSON.stringify(msg) + "\n");
  const respond = (id: unknown, result: unknown) => out({ jsonrpc: "2.0", id, result });
  const respondErr = (id: unknown, code: number, message: string) =>
    out({ jsonrpc: "2.0", id, error: { code, message } });
  const notify = (method: string, params: unknown) => out({ jsonrpc: "2.0", method, params });
  const update = (sessionId: string, u: Record<string, unknown>) =>
    notify("session/update", { sessionId, update: u });

  // Agent → client requests (session/request_permission) get their own ids.
  let nextOutId = 1;
  const pendingClient = new Map<number, (v: any) => void>();
  const clientRequest = (method: string, params: unknown): Promise<any> => {
    const id = nextOutId++;
    return new Promise((resolve) => {
      pendingClient.set(id, resolve);
      out({ jsonrpc: "2.0", id, method, params });
    });
  };

  let permitSeq = 1;

  const modesFor = (config: Config) => ({
    currentModeId: config.mode,
    availableModes: [
      { id: "code", name: "Code", description: "All tools — read, write, run" },
      { id: "plan", name: "Plan", description: "Read-only tools — explore and produce a plan" },
    ],
  });

  const handle = async (msg: any) => {
    // Response to one of OUR requests (permission dialogs).
    if (msg.id != null && msg.method === undefined) {
      const cb = pendingClient.get(msg.id);
      if (cb) { pendingClient.delete(msg.id); cb(msg.error ? null : msg.result); }
      return;
    }
    const { id, method, params } = msg;

    try {
      switch (method) {
        case "initialize": {
          respond(id, {
            protocolVersion: ACP_VERSION,
            agentCapabilities: {
              loadSession: true,
              promptCapabilities: { image: false, audio: false, embeddedContext: true },
            },
            authMethods: [],
          });
          return;
        }
        case "authenticate": {
          respond(id, {});
          return;
        }
        case "session/new": {
          const config = loadConfig({ ...flags, cwd: params?.cwd ?? flags.cwd });
          if (config.permissions === "prompt") config.permissions = "yolo"; // permits flow via ACP instead
          const acpServers = acpMcpSpecs(params?.mcpServers);
          if (Object.keys(acpServers).length) {
            config.mcpServers = { ...config.mcpServers, ...acpServers };
          }
          const provider = resolveProvider(config, flags);
          const session = Session.create(config.dataDir, {
            cwd: config.cwd, model: provider.model, at: new Date().toISOString(),
          });
          config.sessionId = session.id;
          await initMcp(config, (m) => log(m));
          sessions.set(session.id, {
            session, config, provider,
            cp: new Checkpoints(config, session.id),
            ctrl: null, chain: Promise.resolve(),
          });
          log(`session ${session.id} · ${provider.name}/${provider.model} · ${config.cwd}`);
          respond(id, { sessionId: session.id, modes: modesFor(config) });
          return;
        }
        case "session/load": {
          const sessionId = params?.sessionId;
          const config = loadConfig({ ...flags, cwd: params?.cwd ?? flags.cwd });
          if (config.permissions === "prompt") config.permissions = "yolo";
          const provider = resolveProvider(config, flags);
          const session = Session.load(config.dataDir, sessionId);
          config.sessionId = session.id;
          await initMcp(config, (m) => log(m));
          sessions.set(session.id, {
            session, config, provider,
            cp: new Checkpoints(config, session.id),
            ctrl: null, chain: Promise.resolve(),
          });
          for (const m of session.history) {
            if (m.role === "user" && typeof m.content === "string" && !m.content.startsWith("[")) {
              update(session.id, { sessionUpdate: "user_message_chunk", content: { type: "text", text: m.content } });
            } else if (m.role === "assistant" && typeof m.content === "string" && m.content) {
              update(session.id, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: m.content } });
            }
          }
          respond(id, { modes: modesFor(config) });
          return;
        }
        case "session/set_mode": {
          const live = sessions.get(params?.sessionId);
          if (!live) { respondErr(id, -32602, `unknown session: ${params?.sessionId}`); return; }
          if (params?.modeId === "plan" || params?.modeId === "code") {
            live.config.mode = params.modeId;
            update(live.session.id, { sessionUpdate: "current_mode_update", currentModeId: params.modeId });
            respond(id, null);
          } else {
            respondErr(id, -32602, `unknown mode: ${params?.modeId}`);
          }
          return;
        }
        case "session/prompt": {
          const live = sessions.get(params?.sessionId);
          if (!live) { respondErr(id, -32602, `unknown session: ${params?.sessionId}`); return; }
          const text = flattenPrompt(params?.prompt);
          if (!text) { respond(id, { stopReason: "end_turn" }); return; }

          const run = live.chain.then(async () => {
            const ctrl = new AbortController();
            live.ctrl = ctrl;
            const sessionId = live.session.id;
            let mutated = false;
            let maxHit = false;

            const emit = (e: AgentEvent) => {
              switch (e.type) {
                case "text":
                  update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: e.delta } });
                  break;
                case "reasoning":
                  update(sessionId, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: e.delta } });
                  break;
                case "tool_start":
                  update(sessionId, {
                    sessionUpdate: "tool_call",
                    toolCallId: e.id,
                    title: toolLabel(e.name, e.args),
                    kind: TOOL_KIND[e.name] ?? "other",
                    status: "in_progress",
                    rawInput: e.args ?? {},
                  });
                  break;
                case "tool_end":
                  if (!e.error && getTool(e.name)?.readOnly === false) mutated = true;
                  update(sessionId, {
                    sessionUpdate: "tool_call_update",
                    toolCallId: e.id,
                    status: e.error ? "failed" : "completed",
                    content: [{ type: "content", content: { type: "text", text: e.output.slice(0, 4000) } }],
                  });
                  break;
                case "error":
                  if (e.message.startsWith("max iterations")) maxHit = true;
                  log(`error: ${e.message}`);
                  break;
                case "warn":
                  log(`⚠ ${e.message}`);
                  break;
                case "subline":
                  log(e.text);
                  break;
              }
            };

            // Sandbox permits → native editor permission dialogs.
            const permit = async (req: { action: string; detail: string; path?: string }) => {
              const res = await clientRequest("session/request_permission", {
                sessionId,
                toolCall: {
                  toolCallId: `permit-${permitSeq++}`,
                  title: `${req.action}: ${req.path ?? req.detail}`,
                  kind: "other",
                  status: "pending",
                },
                options: [
                  { optionId: "allow", name: "Allow", kind: "allow_once" },
                  { optionId: "reject", name: "Reject", kind: "reject_once" },
                ],
              });
              return res?.outcome?.outcome === "selected" && res.outcome.optionId === "allow";
            };

            try {
              await runTurn(live.config, live.provider, live.session, text, emit, ctrl.signal, { permit });
              if (mutated && live.cp.available()) {
                const hash = live.cp.commit(text);
                if (hash) log(`checkpoint ${hash}`);
              }
              respond(id, { stopReason: ctrl.signal.aborted ? "cancelled" : maxHit ? "max_turn_requests" : "end_turn" });
            } catch (e) {
              if (ctrl.signal.aborted) respond(id, { stopReason: "cancelled" });
              else respondErr(id, -32603, (e as Error).message);
            } finally {
              live.ctrl = null;
            }
          });
          live.chain = run.catch(() => {});
          return;
        }
        case "session/cancel": {
          sessions.get(params?.sessionId)?.ctrl?.abort();
          return; // notification — no response
        }
        default: {
          if (id != null) respondErr(id, -32601, `method not supported: ${method}`);
        }
      }
    } catch (e) {
      if (id != null) respondErr(id, -32603, (e as Error).message);
      else log(`error handling ${method}: ${(e as Error).message}`);
    }
  };

  // Read loop: newline-delimited JSON-RPC on stdin. Handlers run WITHOUT
  // being awaited so session/cancel is processed while a prompt is running.
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of Bun.stdin.stream()) {
    buf += decoder.decode(chunk as Uint8Array, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        void handle(JSON.parse(line));
      } catch {
        log(`unparseable line: ${line.slice(0, 120)}`);
      }
    }
  }
  process.exit(0); // editor closed our stdin
}
