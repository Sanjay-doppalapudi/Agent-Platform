// fetch tool: URL → readable text (crude HTML strip), capped. Zero deps.
// `render:true` is deliberately unavailable: an external browser cannot
// enforce AP's hostname policy across DNS, socket connections, and redirects.
import { isIP } from "node:net";
import { anySignal, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_BYTES = 50_000;

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
    const firstHextet = Number.parseInt(bare.split(":", 1)[0] ?? "", 16);
    if (Number.isFinite(firstHextet) && (firstHextet & 0xffc0) === 0xfe80) {
      return "link-local address";
    }
    const v4mapped = bare.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/i);
    if (v4mapped) return isBlockedFetchHost(v4mapped[1]!);
  }
  return null;
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
    res = await fetch(url, {
      signal: anySignal(ctx.signal, AbortSignal.timeout(15_000)),
      headers: { "user-agent": "ap-agent/1.0", accept: "text/html,text/plain,application/json;q=0.9,*/*;q=0.5" },
      redirect: "follow",
    });
  } catch (e) {
    throw new ToolError(`fetch failed: ${(e as Error).message}`);
  }
  // Re-check the final URL — a redirect onto the metadata endpoint must not
  // slip through after the initial allow.
  try {
    assertFetchUrlAllowed(res.url || url);
  } catch (e) {
    throw e instanceof ToolError ? e : new ToolError(`fetch blocked after redirect`);
  }
  if (!res.ok) throw new ToolError(`fetch failed: HTTP ${res.status} for ${url}`);
  const type = res.headers.get("content-type") ?? "";
  if (/image|video|audio|octet-stream|zip|pdf/.test(type)) {
    return `binary content (${type}) at ${url} — not fetched`;
  }
  const raw = await res.text();
  const text = /html/.test(type) || raw.trimStart().startsWith("<") ? htmlToText(raw) : raw;
  return truncateMiddle(text, MAX_BYTES) || "(empty response)";
}
