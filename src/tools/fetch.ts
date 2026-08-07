// fetch tool: URL → readable text (crude HTML strip), capped. Zero deps.
// render:true runs the page through the SYSTEM's Chrome/Edge in headless mode
// (--dump-dom, one short-lived process) so JS-rendered pages return content;
// nothing is bundled and it degrades to plain fetch when no browser exists.
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
  const h = host.toLowerCase().replace(/^\[|\]$/g, "");
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
    if (bare.startsWith("fe80:")) return "link-local address";
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

// Located once per process; null = probed and absent.
let browserExe: string | null | undefined;

/** Find an installed Chromium-family browser — never download one. */
function findBrowser(): string | null {
  if (browserExe !== undefined) return browserExe;
  const candidates: string[] = [];
  if (process.platform === "win32") {
    const env = process.env;
    for (const base of [env["ProgramFiles"], env["ProgramFiles(x86)"], env["LOCALAPPDATA"]]) {
      if (!base) continue;
      candidates.push(
        join(base, "Google", "Chrome", "Application", "chrome.exe"),
        join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
        join(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
      );
    }
  } else if (process.platform === "darwin") {
    candidates.push(
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
      "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    );
  }
  browserExe = candidates.find((c) => existsSync(c)) ?? null;
  if (!browserExe) {
    for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave"]) {
      const found = Bun.which(name);
      if (found) { browserExe = found; break; }
    }
  }
  return browserExe;
}

/** Rendered DOM via `chrome --headless --dump-dom`, or null if unavailable. */
async function renderWithBrowser(url: string, signal: AbortSignal): Promise<string | null> {
  const exe = findBrowser();
  if (!exe) return null;
  const proc = Bun.spawn(
    [
      exe,
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      "--hide-scrollbars",
      "--mute-audio",
      // Unique per spawn: a shared profile dir is locked by the first Chrome,
      // so concurrent render:true calls in one turn silently failed and were
      // reported as "no browser found". Keep SOME profile dir, though —
      // without one Chrome would attach to the user's real profile.
      `--user-data-dir=${join(tmpdir(), ".ap", `hl-${process.pid}-${Math.random().toString(36).slice(2, 10)}`)}`,
      "--virtual-time-budget=6000", // let JS/fetches settle (virtual clock)
      "--timeout=15000",
      "--dump-dom",
      url,
    ],
    { stdout: "pipe", stderr: "ignore", stdin: "ignore" },
  );
  const timer = setTimeout(() => proc.kill(), 25_000); // hard backstop
  const onAbort = () => proc.kill();
  signal.addEventListener("abort", onAbort, { once: true });
  try {
    const html = await new Response(proc.stdout).text();
    await proc.exited;
    return html.trim().length > 0 ? html : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
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
    const html = await renderWithBrowser(url, ctx.signal);
    if (html !== null) {
      return truncateMiddle(htmlToText(html), MAX_BYTES) || "(page rendered empty)";
    }
    // A cancelled render also returns null. Falling through would fire a
    // brand-new 15s request for a URL the user just cancelled — and blame a
    // missing browser for it.
    if (ctx.signal.aborted) throw new ToolError("fetch cancelled");
    ctx.warn?.("render requested but no Chrome/Edge found — falling back to plain fetch");
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
