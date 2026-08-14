// Artifact tool: the model writes a self-contained HTML page (report,
// diagram, dashboard) to <dataDir>/artifacts/ and the user opens it in a
// browser. Three rules keep it safe: the filename is derived from a
// validated slug and containment-asserted (the skills-installer path
// traversal is the cautionary tale), the page gets a no-network CSP so
// model-authored HTML cannot phone home, and the size is capped like every
// other tool.
import { closeSync, mkdirSync, openSync, writeSync } from "node:fs";
import { join, resolve } from "node:path";
import { ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_HTML_BYTES = 2 * 1024 * 1024; // 2MB — an artifact is a page, not a dataset

/**
 * Meta CSP for generated pages. Scripts and navigation-based exfil are the
 * residual risk with inline script — so script-src is 'none'. form-action /
 * base-uri / connect-src / frame-ancestors are named explicitly (they do not
 * always inherit default-src). Top-level navigation via <meta refresh> is
 * stripped in withCsp; location.href cannot run without script.
 */
const CSP = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; img-src data:; font-src data:; form-action 'none'; base-uri 'none'; connect-src 'none'; frame-ancestors 'none'; object-src 'none'">`;

export function slugify(title: string): string {
  const s = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
  return s || "artifact";
}

/** Strip meta-refresh, scripts, iframes, and inline event handlers; inject CSP. */
export function withCsp(html: string): string {
  let body = html
    .replace(/<meta\b[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*>/gi, "<!-- refresh stripped -->")
    .replace(/<script\b[\s\S]*?<\/script>/gi, "<!-- script stripped -->")
    .replace(/<script\b[^>]*\/>/gi, "<!-- script stripped -->")
    .replace(/<iframe\b[\s\S]*?<\/iframe>/gi, "<!-- iframe stripped -->")
    .replace(/<iframe\b[^>]*\/?>/gi, "<!-- iframe stripped -->")
    // onload= / onclick= / … — CSP blocks script; strip handlers as defense in depth.
    .replace(/\son[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  if (/<head[^>]*>/i.test(body)) return body.replace(/<head([^>]*)>/i, `<head$1>\n${CSP}`);
  if (/<html[^>]*>/i.test(body)) return body.replace(/<html([^>]*)>/i, `<html$1>\n<head>${CSP}</head>`);
  return `<!doctype html>\n<html>\n<head>\n<meta charset="utf-8">\n${CSP}\n</head>\n<body>\n${body}\n</body>\n</html>\n`;
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/** Create exclusively (O_EXCL / wx) so parallel artifact calls never clobber. */
function uniquePathWrite(dir: string, base: string, content: string): string {
  for (let n = 1; n < 1000; n++) {
    const name = n === 1 ? `${base}.html` : `${base}-${n}.html`;
    const p = resolve(dir, name);
    try {
      const fd = openSync(p, "wx");
      try { writeSync(fd, content); } finally { closeSync(fd); }
      return p;
    } catch (e: any) {
      if (e && (e.code === "EEXIST" || e.code === "EPERM")) continue;
      throw e;
    }
  }
  throw new ToolError("could not allocate a unique artifact filename");
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
  const file = uniquePathWrite(dir, `${stamp()}-${slug}`, withCsp(args.html));
  if (!file.startsWith(resolve(dir))) throw new ToolError("artifact path escaped the artifacts directory");
  return `artifact saved: ${file}\nopen it in a browser, or /artifacts lists recent ones${ctx.config.light ? "" : " (ap serve exposes /artifacts/<file>)"}`;
}
