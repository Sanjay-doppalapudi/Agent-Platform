// Minimal HTTP server mode (Bun.serve) — shaped to the call pattern of
// 5Pages' build-server/lib/opencode-runtime.js so an adapter swap is trivial:
//   GET  /health
//   POST /session {cwd?, title?}                → {id}
//   POST /session/:id/message {text, system?, model?, response_format?}
//        blocks until the turn completes        → {text, messages}
//   GET  /event                                 → SSE of all AgentEvents (+sessionId)
//   GET  /session/:id/events                    → SSE scoped to one session
//   GET  /session/:id/messages                  → Msg[]
//   DELETE /session/:id
import { loadConfig, resolveProvider } from "./config.ts";
import { initMcp } from "./mcp.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { Session } from "./session.ts";
import type { CliFlags } from "./index.ts";

const VERSION = "0.1.0";

interface Live {
  session: Session;
  cwd: string;
  chain: Promise<unknown>; // one run at a time per session
  ctrl?: AbortController; // the currently executing run
  closed?: boolean; // DELETE received: refuse new work, let the current run drain
}

export async function serveMain(flags: CliFlags) {
  const baseConfig = loadConfig(flags);
  const provider = resolveProvider(baseConfig, flags);
  const port = flags.port ?? 4141;

  // MCP binds to the server's base cwd once per process; per-session cwds
  // share the same tool set (schema stability across all sessions).
  await initMcp(baseConfig, (m) => console.error(`⚠ ${m}`));

  const sessions = new Map<string, Live>();
  type Subscriber = { sessionId: string | null; write: (e: AgentEvent & { sessionId: string }) => void };
  const subscribers = new Set<Subscriber>();

  const broadcast = (sessionId: string) => (e: AgentEvent) => {
    const evt = { ...e, sessionId };
    for (const sub of subscribers) {
      if (sub.sessionId === null || sub.sessionId === sessionId) sub.write(evt);
    }
  };

  const json = (data: unknown, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

  const sse = (sessionId: string | null): Response => {
    let sub: Subscriber;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        sub = {
          sessionId,
          write: (e) => {
            try { controller.enqueue(enc.encode(`data: ${JSON.stringify(e)}\n\n`)); } catch {}
          },
        };
        subscribers.add(sub);
        controller.enqueue(enc.encode(`: connected\n\n`));
      },
      cancel() {
        subscribers.delete(sub);
      },
    });
    return new Response(stream, {
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      },
    });
  };

  const server = Bun.serve({
    port,
    idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (req.method === "GET" && path === "/health") {
        return json({ ok: true, version: VERSION, provider: provider.name, model: provider.model });
      }
      if (req.method === "GET" && path === "/event") return sse(null);

      if (req.method === "POST" && path === "/session") {
        const body = (await req.json().catch(() => ({}))) as { cwd?: string; title?: string };
        const cwd = body.cwd ?? baseConfig.cwd;
        const session = Session.create(baseConfig.dataDir, {
          cwd,
          model: provider.model,
          at: new Date().toISOString(),
        });
        sessions.set(session.id, { session, cwd, chain: Promise.resolve() });
        return json({ id: session.id });
      }

      const m = path.match(/^\/session\/([^/]+)(?:\/(message|events|messages))?$/);
      if (m) {
        const id = m[1]!;
        const sub = m[2];
        let live = sessions.get(id);
        if (!live) {
          // allow attaching to a persisted session
          try {
            const session = Session.load(baseConfig.dataDir, id);
            live = { session, cwd: baseConfig.cwd, chain: Promise.resolve() };
            sessions.set(id, live);
          } catch {
            return json({ error: "session not found" }, 404);
          }
        }

        if (req.method === "GET" && sub === "events") return sse(id);
        if (req.method === "GET" && sub === "messages") return json(live.session.history);

        if (req.method === "DELETE" && !sub) {
          // Order matters. Detach FIRST (synchronously) so no request can
          // attach to this Live or enqueue onto its chain while we drain;
          // then abort; then wait for the aborted run's final appends
          // (including the cancellation tool results) to land while the file
          // still exists; only then unlink. Deleting first let the running
          // turn's appendFileSync recreate the file as a corrupt, meta-less
          // transcript that then showed up in the resume picker.
          sessions.delete(id);
          live.closed = true;
          live.ctrl?.abort();
          await live.chain.catch(() => {});
          Session.delete(baseConfig.dataDir, id);
          return json({ ok: true });
        }

        if (req.method === "POST" && sub === "message") {
          const body = (await req.json().catch(() => ({}))) as {
            text?: string;
            system?: string;
            model?: string;
            response_format?: unknown;
            allowOutside?: boolean;
          };
          if (!body.text) return json({ error: "text required" }, 400);

          const config = { ...baseConfig, cwd: live.cwd, sessionId: id };
          if (config.permissions === "prompt") config.permissions = "yolo"; // interactive-only
          const prov = body.model ? { ...provider, model: body.model } : provider;
          const extra = body.response_format ? { response_format: body.response_format } : undefined;
          const permit = body.allowOutside ? async () => true : undefined; // undefined → auto-deny

          if (live.closed) return json({ error: "session closed" }, 410);
          const run = live.chain.then(async () => {
            if (live!.closed) throw new Error("session closed"); // queued while deleting
            const ctrl = new AbortController();
            live!.ctrl = ctrl; // visible to DELETE, which must be able to abort it
            const text = await runTurn(
              config, prov, live!.session, body.text!,
              broadcast(id), ctrl.signal,
              { extra, systemOverride: body.system, permit },
            );
            return text;
          });
          live.chain = run.catch(() => {});
          try {
            const text = await run;
            return json({ text, messages: live.session.history });
          } catch (e) {
            return json({ error: (e as Error).message }, 500);
          }
        }
      }

      return json({ error: "not found" }, 404);
    },
  });

  console.log(`AP serving on http://localhost:${server.port} · ${provider.name}/${provider.model}`);
}
