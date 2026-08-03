// fetch tool: URL → readable text (crude HTML strip), capped. Zero deps.
import { truncateMiddle, ToolError } from "./shared.ts";
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

export async function fetchTool(
  args: { url: string },
  _ctx: ToolCtx,
): Promise<string> {
  if (typeof args.url !== "string" || !/^https?:\/\//i.test(args.url)) {
    throw new ToolError("fetch requires {url} starting with http(s)://");
  }
  let res: Response;
  try {
    res = await fetch(args.url, {
      signal: AbortSignal.timeout(15_000),
      headers: { "user-agent": "ap-agent/1.0", accept: "text/html,text/plain,application/json;q=0.9,*/*;q=0.5" },
      redirect: "follow",
    });
  } catch (e) {
    throw new ToolError(`fetch failed: ${(e as Error).message}`);
  }
  if (!res.ok) throw new ToolError(`fetch failed: HTTP ${res.status} for ${args.url}`);
  const type = res.headers.get("content-type") ?? "";
  if (/image|video|audio|octet-stream|zip|pdf/.test(type)) {
    return `binary content (${type}) at ${args.url} — not fetched`;
  }
  const raw = await res.text();
  const text = /html/.test(type) || raw.trimStart().startsWith("<") ? htmlToText(raw) : raw;
  return truncateMiddle(text, MAX_BYTES) || "(empty response)";
}
