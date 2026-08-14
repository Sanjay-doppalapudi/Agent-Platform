// websearch tool: DuckDuckGo's HTML endpoint via plain fetch — no browser,
// no API key, zero deps. Returns "N. Title \n URL \n snippet" blocks.
import { anySignal, ToolError } from "./shared.ts";
import { egressPolicyBlock } from "./fetch.ts";
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
  const egress = egressPolicyBlock(ctx.config.network, "html.duckduckgo.com");
  if (egress) throw new ToolError(`websearch blocked: ${egress}`);
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

  // Parse per RESULT BLOCK so each snippet stays with its own title. Zipping
  // two independently-filtered lists by index desynchronized them the moment
  // one candidate was skipped (an ad), attaching every later snippet to the
  // wrong URL. The split anchors on the container class — `result` followed by
  // a space or quote — so nested `result__*` divs don't fragment a block.
  const results: { url: string; title: string; snippet: string }[] = [];
  for (const block of html.split(/<div class="result[ "]/).slice(1)) {
    const t = block.match(/class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
    if (!t) continue;
    if (/ad_domain=|duckduckgo\.com\/y\.js/.test(t[1]!)) continue; // ad
    const url = realUrl(t[1]!);
    if (!url) continue;
    const s = block.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/);
    results.push({
      url,
      title: decodeEntities(t[2]!),
      snippet: s ? decodeEntities(s[1]!) : "",
    });
    if (results.length >= limit) break;
  }

  if (!results.length) return `no results for "${args.query}"`;
  return results
    .map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}${r.snippet ? `\n   ${r.snippet}` : ""}`)
    .join("\n");
}
