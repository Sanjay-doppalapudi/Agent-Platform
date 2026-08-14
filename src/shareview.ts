// Session share: renders a whole transcript as ONE self-contained HTML file
// (inline CSS, zero external assets) under <dataDir>/shares/. The zero-dep
// answer to hosted share links — host it, mail it, drop it in a PR. Loaded
// lazily; never on the startup path.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Msg } from "./provider.ts";
import { toolLabel } from "./ui.ts";

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Tiny markdown → HTML: headers, fences, bold, inline code, plain lists. */
function mdLite(md: string): string {
  const inline = (s: string) =>
    s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  let html = "";
  let inCode = false;
  let inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of md.split("\n")) {
    if (raw.trim().startsWith("```")) {
      closeList();
      html += inCode ? "</code></pre>" : "<pre><code>";
      inCode = !inCode;
      continue;
    }
    if (inCode) { html += esc(raw) + "\n"; continue; }
    const line = esc(raw);
    const h = line.match(/^(#{1,6})\s+(.*)/);
    const li = line.match(/^\s*(?:[-*+]|\d+\.)\s+(.*)/);
    if (h) {
      closeList();
      html += `<h${Math.min(h[1]!.length + 2, 5)}>${inline(h[2]!)}</h${Math.min(h[1]!.length + 2, 5)}>`;
    } else if (li) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(li[1]!)}</li>`;
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      html += `<p>${inline(line)}</p>`;
    }
  }
  if (inCode) html += "</code></pre>";
  closeList();
  return html;
}

const CLIP = 4000;
const clip = (s: string) => (s.length > CLIP ? `${s.slice(0, CLIP)}\n… [${s.length - CLIP} more chars]` : s);

function renderMsg(m: Msg): string {
  if (m.role === "user" && typeof m.content === "string") {
    const long = m.content.split("\n").length > 14;
    const body = `<div class="body">${mdLite(m.content)}</div>`;
    return `<section class="msg user"><div class="who">you</div>${
      long ? `<details><summary>long message — expand</summary>${body}</details>` : body
    }</section>`;
  }
  if (m.role === "assistant") {
    let out = "";
    if (typeof m.content === "string" && m.content.trim()) {
      out += `<section class="msg agent"><div class="who">agent</div><div class="body">${mdLite(m.content)}</div></section>`;
    }
    for (const tc of (m as any).tool_calls ?? []) {
      let args: any = {};
      try { args = JSON.parse(tc.function?.arguments || "{}"); } catch {}
      out += `<details class="tool"><summary>▸ ${esc(toolLabel(tc.function?.name ?? "?", args))}</summary><pre><code>${esc(clip(JSON.stringify(args, null, 2)))}</code></pre></details>`;
    }
    return out;
  }
  if (m.role === "tool" && typeof (m as any).content === "string") {
    return `<details class="result"><summary>result</summary><pre><code>${esc(clip((m as any).content))}</code></pre></details>`;
  }
  return "";
}

function renderPage(history: Msg[], sessionId: string, model: string, cwd: string): string {
  const turns = history.map(renderMsg).join("");
  return `<!doctype html><html><head><meta charset="utf-8"><title>AP session · ${esc(sessionId)}</title>
<style>
  :root{color-scheme:dark}
  body{background:#0f1115;color:#d7dae0;font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",sans-serif;max-width:860px;margin:0 auto;padding:40px 24px}
  header{border-bottom:1px solid #23262e;padding-bottom:14px;margin-bottom:22px}
  h1{font-size:19px;margin:0 0 4px}
  .meta{color:#6b7280;font-size:12.5px}
  .msg{margin:14px 0;padding:12px 16px;border-radius:10px;border:1px solid #23262e}
  .msg.user{background:#141a26;border-color:#1f2a3d}
  .msg.agent{background:#151920}
  .who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280;margin-bottom:6px}
  .msg.user .who{color:#7aa2f7}
  details.tool,details.result{margin:6px 0 6px 16px;font-size:13px}
  details.tool summary{color:#9aa3b2;cursor:pointer}
  details.result summary{color:#5c6470;cursor:pointer}
  pre{background:#161920;border:1px solid #23262e;border-radius:8px;padding:10px 12px;overflow-x:auto;margin:6px 0}
  code{font:12.5px/1.5 ui-monospace,Consolas,monospace;color:#9ece8f}
  p code,li code{background:#1c212b;padding:1px 5px;border-radius:4px}
  h3,h4,h5{color:#e8eaf0;margin:1.2em 0 .3em}
  ul{margin:.3em 0 .6em}
</style></head><body>
<header>
  <h1>◆ AP session transcript</h1>
  <div class="meta">${esc(sessionId)}${model ? ` · ${esc(model)}` : ""} · ${esc(cwd)} · self-contained file — share it anywhere</div>
</header>
<main>${turns}</main>
</body></html>`;
}

/** Write the transcript page; returns the file path. */
export function exportSessionHtml(
  dataDir: string,
  sessionId: string,
  history: Msg[],
  model: string,
  cwd: string,
): string {
  if (!/^[A-Za-z0-9._-]+$/.test(sessionId) || sessionId.includes("..")) {
    throw new Error(`invalid session id: ${sessionId}`);
  }
  const dir = join(dataDir, "shares");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, `${sessionId}.html`);
  writeFileSync(path, renderPage(history, sessionId, model, cwd));
  return path;
}
