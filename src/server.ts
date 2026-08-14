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
//
// SECURITY: binds 127.0.0.1 by default (Bun's default is 0.0.0.0). Non-loopback
// binds require a bearer token (--token / AP_SERVE_TOKEN); loopback can omit
// it. Unauthenticated network exposure previously meant anyone on the LAN
// could spend the user's API key and run tools as them.
import { loadConfig, resolveProvider, type ResolvedProvider } from "./config.ts";
import { routeTargets } from "./router.ts";
import { initMcp } from "./mcp.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { Session, isValidSessionId } from "./session.ts";
import type { CliFlags } from "./index.ts";
import { timingSafeEqual } from "node:crypto";
import { resolve as resolvePath } from "node:path";

const VERSION = "0.1.15";

function resolveCwd(p: string): string {
  return resolvePath(p);
}

interface Live {
  session: Session;
  cwd: string;
  chain: Promise<unknown>; // one run at a time per session
  ctrl?: AbortController; // the currently executing run
  closed?: boolean; // DELETE received: refuse new work, let the current run drain
}

function isLoopbackHost(host: string): boolean {
  const h = host.toLowerCase();
  return h === "127.0.0.1" || h === "localhost" || h === "::1" || h === "[::1]";
}

export async function serveMain(flags: CliFlags) {
  const baseConfig = loadConfig(flags);
  const provider: ResolvedProvider | ResolvedProvider[] = baseConfig.router?.targets?.length
    ? await routeTargets(baseConfig, flags)
    : resolveProvider(baseConfig, flags);
  const primary = Array.isArray(provider) ? provider[0]! : provider;
  const port = flags.port ?? 4141;
  const hostname = flags.host ?? "127.0.0.1";
  const loopback = isLoopbackHost(hostname);

  let token = flags.token ?? process.env.AP_SERVE_TOKEN ?? "";
  // Always require a bearer token — even on loopback. Local malware / other
  // users on the same machine could otherwise spend the API key and run tools.
  if (!token) {
    token = crypto.randomUUID().replace(/-/g, "");
    console.error(`⚠ ap serve requires a bearer token — generated one (pass Authorization: Bearer <token>)`);
    console.error(`  AP_SERVE_TOKEN=${token}`);
  }
  if (!loopback && !flags.token && !process.env.AP_SERVE_TOKEN) {
    // already generated above; keep the non-loopback note for operators
  }

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

  /** Bearer header only — ?token= leaked into logs, Referer, and process lists. */
  const authorized = (req: Request): boolean => {
    if (!token) return false;
    const h = req.headers.get("authorization") ?? "";
    const want = `Bearer ${token}`;
    if (h.length !== want.length) return false;
    try {
      return timingSafeEqual(Buffer.from(h), Buffer.from(want));
    } catch {
      return false;
    }
  };

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
    hostname,
    idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (req.method === "GET" && path === "/health") {
        // Intentionally does NOT leak provider/model — that advertised which
        // billed account an unauthenticated scanner had found.
        return json({
          ok: true,
          version: VERSION,
          auth: token ? "required" : "off",
          host: hostname,
        });
      }

      if (!authorized(req)) {
        return json({ error: "unauthorized — pass Authorization: Bearer <token>" }, 401);
      }

      if (req.method === "GET" && path === "/event") return sse(null);

      // Generated artifacts (artifact tool). Filename alphabet is validated
      // AND the resolved path is containment-checked — model-influenced names
      // never traverse (the skills installer taught us this lesson).
      const am = path.match(/^\/artifacts\/([A-Za-z0-9][A-Za-z0-9.-]*\.html)$/);
      if (req.method === "GET" && am) {
        const { join, resolve } = await import("node:path");
        const dir = resolve(baseConfig.dataDir, "artifacts");
        const file = resolve(join(dir, am[1]!));
        if (!file.startsWith(dir)) return json({ error: "not found" }, 404);
        const f = Bun.file(file);
        if (!(await f.exists())) return json({ error: "not found" }, 404);
        // CSP header reinforces the in-page meta (and blocks connect-src even
        // if a page strips its own meta). Scripts stay inline-only for the
        // interactive-artifact use case; network egress is denied.
        return new Response(f, {
          headers: {
            "content-type": "text/html; charset=utf-8",
            "content-security-policy":
              "default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; img-src data:; font-src data:; form-action 'none'; base-uri 'none'; connect-src 'none'; frame-ancestors 'none'; object-src 'none'",
            "x-content-type-options": "nosniff",
          },
        });
      }

      if (req.method === "POST" && path === "/session") {
        const body = (await req.json().catch(() => ({}))) as { cwd?: string; title?: string };
        // Remote cwd is a sandbox escape (workspace roots follow session cwd).
        // Sessions always use the server's configured cwd; pass a different
        // root via `ap serve --cwd` on the CLI instead.
        if (body.cwd != null && String(body.cwd).trim() !== "" && resolveCwd(body.cwd) !== resolveCwd(baseConfig.cwd)) {
          return json({
            error: "cwd is not accepted over ap serve — start the server with --cwd, or use the local CLI",
          }, 400);
        }
        const cwd = baseConfig.cwd;
        const session = Session.create(baseConfig.dataDir, {
          cwd,
          model: primary.model,
          at: new Date().toISOString(),
        });
        sessions.set(session.id, { session, cwd, chain: Promise.resolve() });
        return json({ id: session.id });
      }

      const m = path.match(/^\/session\/([^/]+)(?:\/(message|events|messages))?$/);
      if (m) {
        const id = m[1]!;
        if (!isValidSessionId(id)) return json({ error: "session not found" }, 404);
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
          // allowOutside / system from the HTTP body are privilege escalations —
          // ignore them. Operators who need them should use `ap run` locally.
          if (body.allowOutside || body.system) {
            return json({
              error: "allowOutside and system are not accepted over ap serve — use the local CLI",
            }, 400);
          }

          const config = { ...baseConfig, cwd: live.cwd, sessionId: id };
          if (config.permissions === "prompt") config.permissions = "yolo"; // interactive-only
          const prov = Array.isArray(provider)
            ? (body.model ? provider.map((p) => ({ ...p, model: body.model! })) : provider)
            : (body.model ? { ...provider, model: body.model } : provider);
          const extra = body.response_format ? { response_format: body.response_format } : undefined;
          const permit = undefined; // auto-deny outside sandbox

          if (live.closed) return json({ error: "session closed" }, 410);
          const run = live.chain.then(async () => {
            if (live!.closed) throw new Error("session closed"); // queued while deleting
            const ctrl = new AbortController();
            live!.ctrl = ctrl; // visible to DELETE, which must be able to abort it
            const text = await runTurn(
              config, prov, live!.session, body.text!,
              broadcast(id), ctrl.signal,
              { extra, permit },
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

  const authNote = ` · auth required (Authorization: Bearer …)`;
  console.log(`AP serving on http://${hostname === "0.0.0.0" ? "127.0.0.1" : hostname}:${server.port}${authNote} · ${primary.name}/${primary.model}`);
  return server;
}
