// Artifact tool: the model writes a self-contained HTML page (report,
// diagram, dashboard) to <dataDir>/artifacts/ and the user opens it in a
// browser. Three rules keep it safe: the filename is derived from a
// validated slug and containment-asserted (the skills-installer path
// traversal is the cautionary tale), the page gets a no-network CSP so
// model-authored HTML cannot phone home, and the size is capped like every
// other tool.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_HTML_BYTES = 2 * 1024 * 1024; // 2MB — an artifact is a page, not a dataset

/**
 * Meta CSP for generated pages. `default-src 'none'` blocks fetch/XHR/beacon
 * and every remote subresource, but two egress channels do NOT fall back to
 * default-src and must be named explicitly:
 *   form-action — a cross-origin POST form would otherwise submit freely
 *   base-uri    — a rewritten <base> redirects relative targets
 * Top-level navigation (location.href, window.open, meta refresh) cannot be
 * closed by CSP at all, so inline script is the residual risk: keep pages
 * static unless there is a reason not to.
 */
// connect-src + frame-ancestors named explicitly: connect does not always
// inherit default-src in older browsers the way we need, and framing an
// artifact from an attacker page was an easy clickjacking vector.
const CSP = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; font-src data:; form-action 'none'; base-uri 'none'; connect-src 'none'; frame-ancestors 'none'">`;

export function slugify(title: string): string {
  const s = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
  return s || "artifact";
}

/** Inject the CSP into model HTML: after <head>, or wrap when headless. */
export function withCsp(html: string): string {
  if (/<head[^>]*>/i.test(html)) return html.replace(/<head([^>]*)>/i, `<head$1>\n${CSP}`);
  if (/<html[^>]*>/i.test(html)) return html.replace(/<html([^>]*)>/i, `<html$1>\n<head>${CSP}</head>`);
  return `<!doctype html>\n<html>\n<head>\n<meta charset="utf-8">\n${CSP}\n</head>\n<body>\n${html}\n</body>\n</html>\n`;
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/** Seconds-granularity stamps collide when two artifacts land in the same
 *  second (parallel tool calls do), and a plain write would silently destroy
 *  the first. Suffix -2, -3, … until the name is free. */
function uniquePath(dir: string, base: string): string {
  let p = resolve(dir, `${base}.html`);
  for (let n = 2; existsSync(p) && n < 1000; n++) p = resolve(dir, `${base}-${n}.html`);
  return p;
}

export async function artifactTool(
  args: { title: string; html: string; slug?: string },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.title !== "string" || !args.title.trim()) throw new ToolError("artifact requires {title}");
  if (typeof args.html !== "string" || !args.html.trim()) throw new ToolError("artifact requires {html}");
  const bytes = Buffer.byteLength(args.html, "utf8");
  if (bytes > MAX_HTML_BYTES) {
    throw new ToolError(`artifact html is ${Math.round(bytes / 1024)}KB — the cap is ${MAX_HTML_BYTES / 1024}KB; trim embedded data`);
  }
  // The slug is the ONLY model-controlled part of the path. Validate the
  // alphabet, then assert containment anyway — two independent locks.
  const slug = args.slug ? String(args.slug) : slugify(args.title);
  if (!/^[a-z0-9][a-z0-9-]{0,59}$/.test(slug)) {
    throw new ToolError(`artifact slug must match [a-z0-9][a-z0-9-]* (got "${String(args.slug).slice(0, 40)}") — or omit it to derive from the title`);
  }
  const dir = join(ctx.config.dataDir, "artifacts");
  mkdirSync(dir, { recursive: true });
  const file = uniquePath(dir, `${stamp()}-${slug}`);
  if (!file.startsWith(resolve(dir))) throw new ToolError("artifact path escaped the artifacts directory");
  writeFileSync(file, withCsp(args.html));
  return `artifact saved: ${file}\nopen it in a browser, or /artifacts lists recent ones${ctx.config.light ? "" : " (ap serve exposes /artifacts/<file>)"}`;
}
