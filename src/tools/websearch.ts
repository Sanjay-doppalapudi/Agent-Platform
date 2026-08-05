// websearch tool: DuckDuckGo's HTML endpoint via plain fetch — no browser,
// no API key, zero deps. Returns "N. Title \n URL \n snippet" blocks.
import { anySignal, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const decodeEntities = (s: string) =>
  s
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();

/** DDG result links are redirect URLs: //duckduckgo.com/l/?uddg=<encoded>&rut=… */
function realUrl(href: string): string | null {
  const m = href.match(/[?&]uddg=([^&"]+)/);
  if (m) {
    try { return decodeURIComponent(m[1]!); } catch { return null; }
  }
  if (/^https?:\/\//i.test(href)) return href;
  return null;
}

export async function websearchTool(
  args: { query: string; limit?: number },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.query !== "string" || !args.query.trim()) {
    throw new ToolError('websearch requires {query:"search terms"}');
  }
  const limit = Math.min(Math.max(Number(args.limit) || 8, 1), 20);
  let res: Response;
  try {
    res = await fetch("https://html.duckduckgo.com/html/", {
      method: "POST",
      body: new URLSearchParams({ q: args.query.trim() }),
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ap-agent/1.0",
      },
      signal: anySignal(ctx.signal, AbortSignal.timeout(15_000)), // ctrl+c must stop it
      redirect: "follow",
    });
  } catch (e) {
    throw new ToolError(`websearch failed: ${(e as Error).message}`);
  }
  if (!res.ok) throw new ToolError(`websearch failed: HTTP ${res.status}`);
  const html = await res.text();

  // Titles and snippets appear in the same document order — extract each list
  // and zip by index (pairing them in one regex risks spanning result blocks).
  const titles: { url: string; title: string }[] = [];
  for (const m of html.matchAll(/class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)) {
    const url = realUrl(m[1]!);
    if (!url || /ad_domain=|duckduckgo\.com\/y\.js/.test(m[1]!)) continue; // skip ads
    titles.push({ url, title: decodeEntities(m[2]!) });
    if (titles.length >= limit) break;
  }
  const snippets: string[] = [];
  for (const m of html.matchAll(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)) {
    snippets.push(decodeEntities(m[1]!));
    if (snippets.length >= limit) break;
  }

  if (!titles.length) return `no results for "${args.query}"`;
  return titles
    .map((t, i) => `${i + 1}. ${t.title}\n   ${t.url}${snippets[i] ? `\n   ${snippets[i]}` : ""}`)
    .join("\n");
}
