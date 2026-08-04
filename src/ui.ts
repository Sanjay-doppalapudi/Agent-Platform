// Human-friendly tool rendering: labels, result summaries, and diff blocks.
// Everything here is pure string work on data already in memory — no I/O.

const R = "\x1b[0m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";

function trunc(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function lineCount(s: string): number {
  return s ? s.split("\n").filter(Boolean).length : 0;
}

/** One-line human label for a tool call, e.g. `read src/app.ts`. */
export function toolLabel(name: string, args: any): string {
  const a = args ?? {};
  switch (name) {
    case "read": return `read ${a.path ?? ""}${a.offset ? `:${a.offset}` : ""}`;
    case "write": return `write ${a.path ?? ""}`;
    case "edit": return `edit ${a.path ?? ""}`;
    case "bash": return `bash ${trunc(String(a.cmd ?? "").replace(/\s+/g, " "), 70)}${a.background ? " &" : ""}`;
    case "glob": return `glob ${a.pattern ?? ""}`;
    case "grep": return `grep ${trunc(String(a.pattern ?? ""), 40)}${a.glob ? ` (${a.glob})` : ""}${a.mode && a.mode !== "content" ? ` [${a.mode}]` : ""}`;
    case "agent": return `◇ agent ${trunc(String(a.task ?? "").replace(/\s+/g, " "), 60)}`;
    case "fetch": return `fetch ${trunc(String(a.url ?? ""), 60)}`;
    case "todo": return "todo";
    default: return `${name} ${trunc(JSON.stringify(a), 60)}`;
  }
}

/** Short result summary shown after ✓/✗, e.g. `40 lines`. */
export function toolSummary(name: string, output: string, error?: boolean): string {
  if (error) return trunc(output.replace(/^error:\s*/, "").split("\n")[0] ?? "", 80);
  switch (name) {
    case "read": {
      if (output.startsWith("binary file")) return "binary — skipped";
      return `${lineCount(output)} lines`;
    }
    case "glob": return output === "(no matches)" ? "no matches" : `${lineCount(output)} files`;
    case "grep": return output === "(no matches)" ? "no matches" : `${lineCount(output)} hits`;
    case "write": return output.replace(/ to .*$/, ""); // "wrote 123 bytes"
    case "edit": return output.replace(/ in .*$/, "").replace("occurrences", "").replace("occurrence", "").trim();
    case "bash": {
      const first = output.split("\n")[0] ?? "";
      if (first.startsWith("[exit code") || first.startsWith("[timed out")) return trunc(first, 50);
      if (first.startsWith("started background")) return trunc(first.replace("started background process ", ""), 60);
      return output === "(no output)" ? "ok" : `${lineCount(output)} lines out`;
    }
    case "agent": return `${lineCount(output)} lines back`;
    case "fetch": return `${Math.round(Buffer.byteLength(output, "utf8") / 1024)}KB`;
    case "todo": return trunc(output.split("\n")[0] ?? "", 40);
    default: return "";
  }
}

function capLines(lines: string[], max: number): { shown: string[]; hidden: number } {
  if (lines.length <= max) return { shown: lines, hidden: 0 };
  return { shown: lines.slice(0, max), hidden: lines.length - max };
}

// Common provider failures → a suggested next action. Errors are where
// intuition dies; one dim "fix:" line revives it. First match wins.
const ERROR_HINTS: [RegExp, string][] = [
  [/no api key/i, "fix: ap auth <provider>"],
  // Model errors first: some providers wrap them in a 401/403 status.
  [/model\b.*(not (found|supported|available)|does not exist|unknown|invalid)/i, "fix: /models <query> to find a valid id, then /model <provider>/<model>"],
  [/\b401\b|unauthorized|invalid[_ ]?api[_ ]?key|incorrect api key/i, "fix: the key is missing or wrong — ap auth <provider>"],
  [/\b403\b|forbidden/i, "fix: this key can't use that model — /models <query> to find one it can"],
  [/\b429\b|rate[- ]?limit|quota/i, "note: rate-limited — retries are automatic; a cheaper model (-m) also helps"],
  [/\b50[234]\b|overloaded|unavailable/i, "note: provider-side outage — usually brief; /model switches providers"],
  [/idle|stall|timed? ?out/i, "note: the stream stalled — one retry is automatic; check network or /model"],
];

/** Suggested next action for a provider/tool error message, or null. */
export function errorHint(message: string): string | null {
  for (const [re, hint] of ERROR_HINTS) if (re.test(message)) return hint;
  return null;
}

/**
 * Diff block for edit/write, rendered from the tool args (no file reads).
 * For edit: common leading/trailing lines between old and new are trimmed so
 * only the actual change shows. Returns "" for other tools.
 */
export function renderDiff(name: string, args: any, maxWidth?: number): string {
  const width = Math.min((maxWidth ?? process.stdout.columns ?? 80) - 3, 120);
  const out: string[] = [];

  if (name === "edit") {
    const oldRaw = args?.old ?? args?.old_string ?? args?.oldText;
    const newRaw = args?.new ?? args?.new_string ?? args?.newText;
    if (!oldRaw && !newRaw) return ""; // malformed/hallucinated args — no empty block
    const oldL = String(oldRaw ?? "").split("\n");
    const newL = String(newRaw ?? "").split("\n");
    while (oldL.length && newL.length && oldL[0] === newL[0]) { oldL.shift(); newL.shift(); }
    while (oldL.length && newL.length && oldL[oldL.length - 1] === newL[newL.length - 1]) { oldL.pop(); newL.pop(); }
    out.push(`${DIM}${args?.path ?? ""}${R}`);
    const o = capLines(oldL, 12);
    for (const l of o.shown) out.push(`${RED}- ${trunc(l, width)}${R}`);
    if (o.hidden) out.push(`${DIM}  … ${o.hidden} more removed lines${R}`);
    const n = capLines(newL, 12);
    for (const l of n.shown) out.push(`${GREEN}+ ${trunc(l, width)}${R}`);
    if (n.hidden) out.push(`${DIM}  … ${n.hidden} more added lines${R}`);
    return out.join("\n") + "\n";
  }

  if (name === "write") {
    const lines = String(args?.content ?? "").split("\n");
    out.push(`${DIM}${args?.path ?? ""} · new file · ${lines.length} lines${R}`);
    const c = capLines(lines, 12);
    for (const l of c.shown) out.push(`${GREEN}+ ${trunc(l, width)}${R}`);
    if (c.hidden) out.push(`${DIM}  … ${c.hidden} more lines${R}`);
    return out.join("\n") + "\n";
  }

  return "";
}
