// fetch tool: URL → readable text (crude HTML strip), capped. Zero deps.
// `render:true` is deliberately unavailable: an external browser cannot
// enforce AP's hostname policy across DNS, socket connections, and redirects.
import { lookup } from "node:dns/promises";
import { request as httpRequest, type IncomingMessage } from "node:http";
import { request as httpsRequest } from "node:https";
import { isIP } from "node:net";
import { Readable } from "node:stream";
import { anySignal, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_BYTES = 50_000;
const MAX_REDIRECTS = 10;
const FETCH_HEADERS = {
  "user-agent": "ap-agent/1.0",
  accept: "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
};

function htmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|h[1-6]|li|tr|section|article)>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Block cloud-metadata / link-local targets. Localhost and RFC1918 stay
 * allowed — coding agents routinely fetch local swagger/docs — but the
 * well-known IMDSv1/v2 and GCP metadata endpoints are never legitimate
 * documentation URLs for a coding task.
 */
export function isBlockedFetchHost(host: string): string | null {
  const unbracketed = host.toLowerCase().replace(/^\[|\]$/g, "");
  // A trailing DNS root dot is semantically equivalent to the bare hostname.
  const h = unbracketed.endsWith(".") ? unbracketed.slice(0, -1) : unbracketed;
  if (h === "metadata.google.internal" || h.endsWith(".metadata.google.internal")) {
    return "cloud metadata host";
  }
  // Strip zone id (fe80::1%eth0) before parsing.
  const bare = h.split("%")[0]!;
  const ipVersion = isIP(bare);
  if (ipVersion === 4) {
    const parts = bare.split(".").map(Number);
    if (parts[0] === 169 && parts[1] === 254) return "link-local / cloud metadata address";
    if (parts[0] === 0) return "unspecified address";
  } else if (ipVersion === 6) {
    // fd00:ec2::254 (AWS IMDS), fe80::/10 link-local, ::ffff:169.254.x.x
    if (bare === "fd00:ec2::254" || bare.startsWith("fd00:ec2:")) return "cloud metadata address";
    const hx = hextets(bare);
    if (hx) {
      if ((hx[0]! & 0xffc0) === 0xfe80) return "link-local address";
      // An IPv4 address embedded in IPv6 reaches the SAME host on a
      // dual-stack socket, so it must face the same policy. Checked
      // numerically: new URL() re-serializes [::ffff:169.254.169.254] as
      // [::ffff:a9fe:a9fe], so a dotted-quad regex alone never fires on
      // anything that came through URL parsing.
      const v4 = embeddedIPv4(hx);
      if (v4) return isBlockedFetchHost(v4);
    }
    // Kept for the dotted spelling, which dns.lookup can still return.
    const v4mapped = bare.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/i);
    if (v4mapped) return isBlockedFetchHost(v4mapped[1]!);
  }
  return null;
}

/** Expand an IPv6 literal (with optional "::") into 8 numeric hextets. */
function hextets(ip: string): number[] | null {
  if (ip.includes(".")) return null; // dotted tail — handled by the regex branch
  const halves = ip.split("::");
  if (halves.length > 2) return null;
  const head = halves[0] ? halves[0].split(":") : [];
  const tail = halves.length === 2 ? (halves[1] ? halves[1].split(":") : []) : null;
  let parts: string[];
  if (tail === null) parts = head;
  else {
    const fill = 8 - head.length - tail.length;
    if (fill < 0) return null;
    parts = [...head, ...Array(fill).fill("0"), ...tail];
  }
  if (parts.length !== 8) return null;
  const out = parts.map((p) => Number.parseInt(p, 16));
  return out.every((n) => Number.isFinite(n) && n >= 0 && n <= 0xffff) ? out : null;
}

/** Dotted IPv4 embedded in an IPv6 address: ::ffff:a.b.c.d (v4-mapped),
 *  ::a.b.c.d (v4-compatible), 64:ff9b::a.b.c.d (NAT64). */
function embeddedIPv4(x: number[]): string | null {
  const zeros5 = x[0] === 0 && x[1] === 0 && x[2] === 0 && x[3] === 0 && x[4] === 0;
  const mapped = zeros5 && x[5] === 0xffff;
  const compat = zeros5 && x[5] === 0 && x[6] !== 0; // excludes :: and ::1
  const nat64 = x[0] === 0x64 && x[1] === 0xff9b && x[2] === 0 && x[3] === 0 && x[4] === 0 && x[5] === 0;
  if (!(mapped || compat || nat64)) return null;
  return [x[6]! >>> 8, x[6]! & 255, x[7]! >>> 8, x[7]! & 255].join(".");
}

export function assertFetchUrlAllowed(raw: string): URL {
  let u: URL;
  try { u = new URL(raw); } catch { throw new ToolError(`fetch requires a valid http(s) URL`); }
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    throw new ToolError("fetch requires {url} starting with http(s)://");
  }
  const why = isBlockedFetchHost(u.hostname);
  if (why) throw new ToolError(`fetch blocked: ${why} (${u.hostname})`);
  return u;
}

/** Resolve once, validate the address, then pin the HTTP connection to it. */
async function resolveAllowedAddress(hostname: string): Promise<{ address: string; family: number }> {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
  const family = isIP(host);
  const addresses = family
    ? [{ address: host, family }]
    : await lookup(host, { all: true, verbatim: true });
  const allowed = addresses.find(({ address }) => !isBlockedFetchHost(address));
  if (!allowed) {
    const address = addresses[0]?.address ?? hostname;
    throw new ToolError(`fetch blocked: ${isBlockedFetchHost(address) ?? "unsafe resolved address"} (${address})`);
  }
  return allowed;
}

function responseFromIncoming(incoming: IncomingMessage): Response {
  const headers = new Headers();
  for (const [name, value] of Object.entries(incoming.headers)) {
    if (Array.isArray(value)) for (const item of value) headers.append(name, item);
    else if (value !== undefined) headers.set(name, value);
  }
  return new Response(Readable.toWeb(incoming) as ReadableStream<Uint8Array>, {
    status: incoming.statusCode ?? 500,
    headers,
  });
}

async function requestOnce(url: URL, signal: AbortSignal): Promise<{ incoming: IncomingMessage; response: Response }> {
  const { address, family } = await resolveAllowedAddress(url.hostname);
  if (signal.aborted) throw signal.reason ?? new Error("fetch cancelled");
  return new Promise((resolve, reject) => {
    const request = (url.protocol === "https:" ? httpsRequest : httpRequest) as typeof httpRequest;
    const req = request(url, {
      headers: FETCH_HEADERS,
      signal,
      // Pin the already-vetted DNS result. Without this callback, a hostname
      // can rebind between validation and connect.
      //
      // net.Socket calls lookup with {all: true} and then sorts the result,
      // so the callback MUST hand back an ARRAY in that mode. Replying with
      // the 3-arg (err, address, family) form made every hostname fetch die
      // with "results.sort is not a function" — literal-IP URLs still worked,
      // which is why it was not obvious.
      lookup: (
        _host: string,
        opts: { all?: boolean } | undefined,
        done: (error: Error | null, address: string | { address: string; family: number }[], family?: number) => void,
      ) => (opts?.all ? done(null, [{ address, family }]) : done(null, address, family)),
    } as any, (incoming) => resolve({ incoming, response: responseFromIncoming(incoming) }));
    req.once("error", reject);
    req.end();
  });
}

/** Follow redirects manually so every target is validated before connecting. */
async function safeFetch(raw: string, signal: AbortSignal): Promise<Response> {
  let url = assertFetchUrlAllowed(raw);
  for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects++) {
    const { incoming, response } = await requestOnce(url, signal);
    const status = incoming.statusCode ?? 0;
    const location = incoming.headers.location;
    if (location && [301, 302, 303, 307, 308].includes(status)) {
      incoming.resume(); // discard this redirect body before the next request
      url = assertFetchUrlAllowed(new URL(location, url).href);
      continue;
    }
    return response;
  }
  throw new ToolError(`fetch failed: too many redirects (max ${MAX_REDIRECTS})`);
}

export async function fetchTool(
  args: { url: string; render?: boolean },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.url !== "string") {
    throw new ToolError("fetch requires {url} starting with http(s)://");
  }
  const url = assertFetchUrlAllowed(args.url).href;
  if (args.render) {
    throw new ToolError(
      "fetch render:true is temporarily disabled: the system browser cannot enforce the hostname policy across redirects and socket connections. Use plain fetch or a sandboxed browser.",
    );
  }
  let res: Response;
  try {
    res = await safeFetch(url, anySignal(ctx.signal, AbortSignal.timeout(15_000)));
  } catch (e) {
    throw e instanceof ToolError ? e : new ToolError(`fetch failed: ${(e as Error).message}`);
  }
  if (!res.ok) throw new ToolError(`fetch failed: HTTP ${res.status} for ${url}`);
  const type = res.headers.get("content-type") ?? "";
  if (/image|video|audio|octet-stream|zip|pdf/.test(type)) {
    return `binary content (${type}) at ${url} — not fetched`;
  }
  // Read a BOUNDED prefix. res.text() buffers whatever the server sends, so a
  // hostile (or merely huge) URL could pull gigabytes into memory before the
  // MAX_BYTES truncation ever ran. Stop at a small multiple of the cap —
  // enough that HTML stripping still has material to work with.
  const raw = await readCapped(res, MAX_BYTES * 4);
  const text = /html/.test(type) || raw.trimStart().startsWith("<") ? htmlToText(raw) : raw;
  return truncateMiddle(text, MAX_BYTES) || "(empty response)";
}

/** Decode at most `limit` bytes of a response, then drop the connection. */
async function readCapped(res: Response, limit: number): Promise<string> {
  const body = res.body;
  if (!body) return "";
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      chunks.push(value);
      total += value.byteLength;
      if (total >= limit) break;
    }
  } finally {
    try { await reader.cancel(); } catch {}
  }
  const buf = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.byteLength; }
  return new TextDecoder().decode(buf.subarray(0, Math.min(total, limit)));
}
