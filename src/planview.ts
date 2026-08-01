// Plan export: renders a plan as a self-contained interactive HTML page
// (checkable steps + progress bar) in the OS temp dir and opens it in the
// default browser. Loaded lazily — never on the startup path.
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const WORDS = [
  "ace", "ant", "arc", "ash", "bat", "bee", "box", "bug", "cab", "cat",
  "cod", "cog", "cup", "dew", "dot", "elk", "elm", "fig", "fin", "fox",
  "gem", "gnu", "hat", "hen", "ice", "ink", "jam", "jet", "kit", "koi",
  "lab", "log", "map", "mud", "oak", "orb", "owl", "pan", "peg", "pin",
  "ram", "ray", "rig", "sap", "sky", "tab", "tin", "urn", "van", "wax",
];

function randomName(): string {
  const pick = () => WORDS[Math.floor(Math.random() * WORDS.length)]!;
  return `${pick()}_${pick()}_${pick()}`;
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Tiny markdown → HTML: headers, fences, lists (as checkable steps), bold, inline code. */
function mdToHtml(md: string): string {
  const inline = (s: string) =>
    s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  let html = "";
  let inCode = false;
  let inList = false;
  const closeList = () => { if (inList) { html += "</ol>"; inList = false; } };
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
      const lvl = Math.min(h[1]!.length + 1, 4);
      html += `<h${lvl}>${inline(h[2]!)}</h${lvl}>`;
    } else if (li) {
      if (!inList) { html += `<ol class="steps">`; inList = true; }
      html += `<li onclick="this.classList.toggle('done');prog()">${inline(li[1]!)}</li>`;
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

function renderPage(plan: string, model: string, cwd: string, title: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><title>plan · ${esc(title)}</title>
<style>
  :root{color-scheme:dark}
  body{background:#0f1115;color:#d7dae0;font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",sans-serif;max-width:820px;margin:0 auto;padding:40px 24px}
  header{border-bottom:1px solid #23262e;padding-bottom:14px;margin-bottom:22px}
  h1{font-size:20px;margin:0 0 4px}
  h2,h3,h4{color:#e8eaf0;margin:1.4em 0 .4em}
  .meta{color:#6b7280;font-size:12.5px}
  .bar{height:6px;background:#23262e;border-radius:3px;margin-top:12px;overflow:hidden}
  .bar i{display:block;height:100%;width:0;background:#4ade80;transition:width .2s}
  #count{color:#4ade80}
  ol.steps{list-style:none;counter-reset:s;padding:0}
  ol.steps li{counter-increment:s;padding:9px 12px 9px 44px;margin:6px 0;background:#161920;border:1px solid #23262e;border-radius:8px;position:relative;cursor:pointer;user-select:none}
  ol.steps li:hover{border-color:#3b4252}
  ol.steps li::before{content:counter(s);position:absolute;left:12px;top:8px;width:22px;height:22px;border-radius:50%;background:#23262e;color:#9aa3b2;font-size:12px;display:flex;align-items:center;justify-content:center}
  ol.steps li.done{opacity:.45;text-decoration:line-through}
  ol.steps li.done::before{content:"✓";background:#14532d;color:#4ade80}
  pre{background:#161920;border:1px solid #23262e;border-radius:8px;padding:12px 14px;overflow-x:auto}
  code{font:13px/1.5 ui-monospace,Consolas,monospace;color:#9ece8f}
  p code,li code{background:#1c212b;padding:1px 5px;border-radius:4px}
  button{background:#1c212b;color:#d7dae0;border:1px solid #2d3340;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:12.5px;float:right}
  button:hover{border-color:#4b5563}
</style></head><body>
<header>
  <button onclick="navigator.clipboard.writeText(document.getElementById('raw').textContent).then(()=>this.textContent='copied!')">copy plan</button>
  <h1>◆ AP plan · ${esc(title)}</h1>
  <div class="meta">${esc(model)} · ${esc(cwd)} · click a step to mark it done — <span id="count"></span></div>
  <div class="bar"><i id="bar"></i></div>
</header>
<main>${mdToHtml(plan)}</main>
<script id="raw" type="text/plain">${esc(plan)}</script>
<script>
function prog(){
  const li=[...document.querySelectorAll("ol.steps li")],d=li.filter(x=>x.classList.contains("done")).length;
  document.getElementById("bar").style.width=li.length?(100*d/li.length)+"%":"0";
  document.getElementById("count").textContent=li.length?d+"/"+li.length+" steps done":"no checkable steps";
}
prog();
</script></body></html>`;
}

export function exportPlanHtml(plan: string, model: string, cwd: string, sessionId: string): string {
  const dir = join(tmpdir(), ".ap", sessionId);
  mkdirSync(dir, { recursive: true });
  const name = randomName();
  const path = join(dir, `${name}.html`);
  writeFileSync(path, renderPage(plan, model, cwd, name));
  return path;
}

export function openInBrowser(path: string) {
  const cmd =
    process.platform === "win32" ? ["cmd", "/c", "start", "", path] :
    process.platform === "darwin" ? ["open", path] :
    ["xdg-open", path];
  try {
    Bun.spawn(cmd, { stdout: "ignore", stderr: "ignore", stdin: "ignore" }).unref();
  } catch {}
}
