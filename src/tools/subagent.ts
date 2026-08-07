// The child-process subagent runner, extracted from the agent tool so the
// same machinery serves three callers: the synchronous agent tool, background
// task delegation (agent background:true), and workflow scripts (flow.ts).
// Children are always `ap run --json --light`, so they cannot spawn further
// agents — the structural recursion cap lives in the profile, not in code.
import { existsSync } from "node:fs";

/** How to re-invoke ourselves: compiled exe vs `bun src/index.ts`.
 * Compiled binaries embed the entry at a VIRTUAL path — "/$bunfs/…" on
 * unix, "B:\~BUN\…" on Windows — so the reliable test is: does the entry
 * exist on disk? If not, we're compiled and the exe alone re-runs itself. */
export function selfCmd(): string[] {
  const entry = process.argv[1] ?? "";
  if (!entry || entry.includes("$bunfs") || entry.includes("~BUN") || !existsSync(entry)) {
    return [process.execPath];
  }
  return [process.execPath, entry];
}

export interface SubagentResult {
  text: string;
  steps: number;
  killed: boolean;
  exitCode: number | null;
  /** Non-retryable error event from the child, if any. */
  errorMsg: string;
  /** Last ~1KB of child stderr — the only place startup crashes surface. */
  stderrTail: string;
}

export interface SubagentOpts {
  task: string;
  cwd: string;
  timeoutMs: number;
  /** Extra argv for the child (e.g. ["--agent", "reviewer"]). */
  extraArgs?: string[];
  /** Abort hook — background tasks pass their own controller, never the
   *  turn's signal (a background task must survive the turn that started it). */
  signal?: AbortSignal;
  onStep?: (line: string) => void;
  /** Live text/reasoning deltas from the child, as they stream. The child
   *  already emits them over NDJSON; without this hook they were discarded,
   *  so a running subagent looked like a black box until it finished. */
  onOutput?: (kind: "reasoning" | "text", delta: string) => void;
}

/** Spawn one `ap run --json --light` child and collect its event stream. */
export async function runSubagent(opts: SubagentOpts): Promise<SubagentResult> {
  const proc = Bun.spawn(
    [...selfCmd(), "run", "-p", opts.task, "--json", "--light", "--cwd", opts.cwd, ...(opts.extraArgs ?? [])],
    { stdout: "pipe", stderr: "pipe", stdin: "ignore", windowsHide: true } as any,
  );
  let stderrTail = "";
  const stderrDone = (async () => {
    try {
      for await (const chunk of proc.stderr as any) {
        stderrTail = (stderrTail + new TextDecoder().decode(chunk)).slice(-1000);
      }
    } catch {}
  })();

  let killed = false;
  const timer = setTimeout(() => { killed = true; proc.kill(); }, opts.timeoutMs);
  const onAbort = () => { killed = true; proc.kill(); };
  opts.signal?.addEventListener("abort", onAbort, { once: true });

  let text = "";
  let errorMsg = "";
  let steps = 0;
  try {
    const reader = proc.stdout.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        try {
          const e = JSON.parse(line);
          if (e.type === "reasoning" || e.type === "text") {
            if (typeof e.delta === "string" && e.delta) opts.onOutput?.(e.type, e.delta);
          } else if (e.type === "tool_end") {
            steps++;
            opts.onStep?.(`  ↳ ${e.error ? "✗" : "✓"} ${e.name} · ${e.ms}ms`);
          } else if (e.type === "done") {
            text = e.text ?? "";
          } else if (e.type === "error" && !e.retryable) {
            errorMsg = e.message ?? "";
          }
        } catch {}
      }
    }
    await proc.exited;
    await stderrDone;
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
  }
  return { text, steps, killed, exitCode: proc.exitCode, errorMsg, stderrTail };
}
