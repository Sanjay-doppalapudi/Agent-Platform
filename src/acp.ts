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
import { initMcp, mcpStatus } from "./mcp.ts";
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

// Slash commands advertised to the editor (available_commands_update).
// Zed blocks any command NOT in this list client-side, so this is the
// complete surface; invocations arrive as session/prompt text "/name args".
const SLASH_COMMANDS = [
  { name: "plan", description: "Read-only mode: explore and produce a plan" },
  { name: "code", description: "Full mode: all tools (default)" },
  { name: "model", description: "Switch model, or provider/model", input: { hint: "provider/model" } },
  { name: "undo", description: "Restore the previous checkpoint" },
  { name: "diff", description: "Diff of the last checkpoint (+ pending changes)" },
  { name: "checkpoints", description: "List workspace checkpoints" },
  { name: "mcp", description: "List MCP servers and their tools" },
  { name: "skills", description: "List available SKILL.md packs" },
  { name: "context", description: "Show session context usage" },
];

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

/** Execute an advertised slash command against a live session. */
async function runCommand(live: AcpLive, name: string, arg: string): Promise<string> {
  switch (name) {
    case "plan":
    case "code":
      live.config.mode = name;
      return `mode: ${name}${name === "plan" ? " — read-only tools, mutations blocked" : " — all tools"}`;
    case "model": {
      if (!arg) return `current model: ${live.provider.name}/${live.provider.model}`;
      const slash = arg.indexOf("/");
      const prov = slash > 0 ? arg.slice(0, slash) : "";
      if (prov && live.config.providers[prov]) {
        live.provider = resolveProvider({ ...live.config, provider: prov }, { model: arg.slice(slash + 1) } as CliFlags);
      } else {
        live.provider = { ...live.provider, model: arg };
      }
      return `model: ${live.provider.name}/${live.provider.model}`;
    }
    case "undo": {
      const cps = live.cp.list(2);
      if (cps.length < 2) return "no earlier checkpoint to restore";
      const r = live.cp.restore(cps[1]!.hash);
      return r ? `restored checkpoint ${cps[1]!.hash} (${cps[1]!.label})` : "restore failed";
    }
    case "diff":
      return live.cp.available() ? live.cp.diff(1, 20_000) : "(checkpoints unavailable)";
    case "checkpoints": {
      const cps = live.cp.list();
      return cps.length ? cps.map((c) => `${c.hash}  ${c.label}`).join("\n") : "no checkpoints yet";
    }
    case "mcp": {
      const st = mcpStatus();
      if (!st.length) return "no MCP servers connected — configure them in .mcp.json or ap.config.json";
      return st.map((s) =>
        `${s.ok ? "✓" : "✗"} ${s.name} [${s.transport}] ${s.ok ? `· ${s.tools.length} tools: ${s.tools.map((t) => t.name).join(", ").slice(0, 200)}` : `· ${s.error}`}`,
      ).join("\n");
    }
    case "skills": {
      const { discoverSkills } = await import("./skills.ts");
      const sk = discoverSkills(live.config);
      return sk.length
        ? sk.map((s) => `${s.name} [${s.source}] ${s.description.slice(0, 80)}`).join("\n")
        : "no skills installed — ap skills add <owner>/<repo>";
    }
    case "context": {
      const chars = live.session.history.reduce((n, m) => n + JSON.stringify(m).length, 0);
      return `messages: ${live.session.history.length} · ~${Math.round(chars / 4)} tokens (${chars} chars) · trim budget ${live.config.contextBudgetChars} chars`;
    }
    default:
      return `unknown command: /${name}`;
  }
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
  const advertiseCommands = (sessionId: string) =>
    update(sessionId, { sessionUpdate: "available_commands_update", availableCommands: SLASH_COMMANDS });

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
          advertiseCommands(session.id);
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
          advertiseCommands(session.id);
          return;
        }
        case "session/set_mode": {
          const live = sessions.get(params?.sessionId);
          if (!live) { respondErr(id, -32602, `unknown session: ${params?.sessionId}`); return; }
          if (params?.modeId === "plan" || params?.modeId === "code") {
            live.config.mode = params.modeId;
            // Result must be an OBJECT — Zed's client rejects null and
            // reverts the mode picker on deserialization failure.
            respond(id, {});
            update(live.session.id, { sessionUpdate: "current_mode_update", currentModeId: params.modeId });
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

          // Advertised slash commands arrive as prompt text — handle locally,
          // no model round-trip.
          const cmd = text.match(/^\/([a-z]+)\s*([\s\S]*)$/);
          if (cmd && SLASH_COMMANDS.some((c) => c.name === cmd[1])) {
            const result = await runCommand(live, cmd[1]!, cmd[2]!.trim());
            update(live.session.id, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: result } });
            if (cmd[1] === "plan" || cmd[1] === "code") {
              update(live.session.id, { sessionUpdate: "current_mode_update", currentModeId: cmd[1] });
            }
            respond(id, { stopReason: "end_turn" });
            return;
          }

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
