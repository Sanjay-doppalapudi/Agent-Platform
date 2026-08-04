// `ap loop -p "goal"` — autonomous completion loop. Runs work iterations
// until the goal is VERIFIABLY done, not merely claimed done:
//
//   work turn → objective gates (--check commands, zero tokens) →
//   verifier audit turn → LOOP_DONE + clean audit? stop : feed gaps back.
//
// Efficiency levers (this is why it beats a "keep going" prompt):
//   - checks run BEFORE any verifier tokens are spent; a failing check
//     short-circuits straight back to work with the command output
//   - one append-only session → provider prefix cache hits every iteration
//   - fixed-byte verifier prompt (cacheable), audit doubles as next task list
//   - stall detector: two identical audits with no file mutations → stop
//     instead of burning tokens forever
//   - auto-compaction: near the context budget the session is summarized
//     into a fresh one (goal + progress), so the loop can run indefinitely
//   - checkpoint per mutating iteration → /undo-able audit trail
import { loadConfig, resolveProvider } from "./config.ts";
import { initMcp } from "./mcp.ts";
import { runTurn, type AgentEvent } from "./agent.ts";
import { Checkpoints } from "./checkpoint.ts";
import { getTool } from "./tools/index.ts";
import { shellPrefix } from "./tools/bash.ts";
import { MdRenderer } from "./md.ts";
import { renderDiff, toolLabel, toolSummary } from "./ui.ts";
import { Session } from "./session.ts";
import type { CliFlags } from "./index.ts";

const VERIFY_PROMPT = `[loop verifier] STOP working and audit. Re-read the ORIGINAL GOAL at the top of this session. For each requirement, verify it concretely NOW using ONLY the read/grep/glob tools — do not use bash, do not trust your memory of earlier turns. Then reply with exactly LOOP_DONE if every requirement is met, or a terse numbered list of what is still missing or broken, or LOOP_BLOCKED: <reason> if the goal does not apply to this directory at all.`;

const CONTINUE_PROMPT = `[loop] Continue: complete every unmet item from your audit above. Do the work now.`;

const COMPACT_PROMPT = `[loop] Handoff summary for a fresh context, terse: 1) DONE so far 2) REMAINING 3) key files/paths 4) gotchas discovered. No prose padding.`;

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

export async function loopMain(flags: CliFlags) {
  if (!flags.prompt) {
    console.error(`usage: ap loop -p "goal" [--check "cmd"]... [--max N] [--json]`);
    process.exit(1);
  }
  const config = loadConfig(flags);
  if (config.permissions === "prompt") config.permissions = "yolo"; // headless
  const provider = resolveProvider(config, flags);
  const checks = flags.check ?? [];
  const maxIter = flags.max ?? 0; // 0 = unlimited — ctrl+c / stall / done stop it
  const json = !!flags.json;

  let session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
  config.sessionId = session.id;

  const ctrl = new AbortController();
  process.on("SIGINT", () => ctrl.abort());

  await initMcp(config, (m) => process.stderr.write(`\x1b[33m⚠ ${m}\x1b[0m\n`));

  // --- rendering (same contract as run mode; loop meta-lines on stderr) ----
  const md = !json && process.stdout.isTTY ? new MdRenderer() : null;
  const active = new Map<string, string>();
  let sawText = false;
  let mutatedThisTurn = false;
  let lastText = "";
  const status = (line: string) => {
    if (json) process.stdout.write(JSON.stringify({ type: "loop", message: line }) + "\n");
    else process.stderr.write(`\x1b[36m◐ ${line}\x1b[0m\n`);
  };
  const emit = (e: AgentEvent) => {
    if (e.type === "text") lastText += e.delta;
    if (e.type === "tool_end" && !e.error && getTool(e.name)?.readOnly === false) mutatedThisTurn = true;
    if (json) { process.stdout.write(JSON.stringify(e) + "\n"); return; }
    switch (e.type) {
      case "reasoning": process.stderr.write(`\x1b[2m${e.delta}\x1b[0m`); break;
      case "warn": process.stderr.write(`\x1b[33m⚠ ${e.message}\x1b[0m\n`); break;
      case "subline": process.stderr.write(`\x1b[2m  ${e.text}\x1b[0m\n`); break;
      case "text":
        process.stdout.write(md ? md.push(e.delta) : e.delta);
        sawText = true;
        break;
      case "tool_start": {
        if (sawText) { if (md) process.stdout.write(md.flush()); process.stdout.write("\n"); sawText = false; }
        active.set(e.id, toolLabel(e.name, e.args));
        const diff = renderDiff(e.name, e.args);
        if (diff) process.stderr.write(diff);
        break;
      }
      case "tool_end": {
        const label = active.get(e.id) ?? e.name;
        active.delete(e.id);
        process.stderr.write(`${e.error ? "✗" : "✓"} ${label} · ${toolSummary(e.name, e.output, e.error)} · ${e.ms}ms\n`);
        break;
      }
      case "error": process.stderr.write(`error: ${e.message}\n`); break;
    }
  };

  const turn = async (userText: string): Promise<{ text: string; mutated: boolean }> => {
    lastText = ""; mutatedThisTurn = false;
    await runTurn(config, provider, session, userText, emit, ctrl.signal, {
      permit: flags.allowOutside ? async () => true : undefined,
    });
    if (md) process.stdout.write(md.flush());
    if (!json) process.stdout.write("\n");
    return { text: lastText, mutated: mutatedThisTurn };
  };

  const runChecks = (): { cmd: string; out: string } | null => {
    for (const cmd of checks) {
      const p = Bun.spawnSync([...shellPrefix(config.shell), cmd], {
        cwd: config.cwd, stdout: "pipe", stderr: "pipe", timeout: 600_000,
      });
      if (p.exitCode !== 0) {
        const out = `${p.stdout?.toString() ?? ""}\n${p.stderr?.toString() ?? ""}`.trim();
        return { cmd, out: out.slice(-2000) };
      }
    }
    return null;
  };

  let cp = new Checkpoints(config, session.id);
  let loopStart: string | null = cp.available() ? cp.head() : null;
  const checkpoint = (label: string) => {
    if (cp.available()) {
      const hash = cp.commit(label);
      if (hash && !json) process.stderr.write(`\x1b[2m✓ checkpoint ${hash}\x1b[0m\n`);
    }
  };

  // Collapsed per-file diff summary (name + added/removed lines), printed
  // after every iteration and cumulatively when the loop ends.
  const showChanges = (from: string | null, scope: string) => {
    if (!cp.available()) return;
    const files = cp.filesChanged(from);
    if (!files.length) return;
    if (json) {
      process.stdout.write(JSON.stringify({ type: "loop_diff", scope, files }) + "\n");
      return;
    }
    const w = Math.min(60, Math.max(...files.map((f) => f.file.length)));
    const lines = files.map((f) => `    ${f.file.padEnd(w)}  \x1b[32m+${f.add}\x1b[0m \x1b[31m-${f.del}\x1b[0m`);
    process.stderr.write(`\x1b[2m  ▸ changes (${scope})\x1b[0m\n${lines.join("\n")}\n`);
  };

  const budget = Math.floor((config.contextBudgetChars ?? 400_000) * 0.6);
  const historySize = () => session.history.reduce((n, m) => n + JSON.stringify(m).length, 0);

  const goal = flags.prompt;
  let nextMsg = `[loop mode] GOAL:\n${goal}\n\nWork until this goal is FULLY complete. Claims are not enough — an auditor re-verifies everything, and any verification command must pass. You will be re-invoked until it does.\nIf the goal does not apply to this directory (e.g. it mentions tests and none exist here), do NOT hunt for meaning in other folders — reply LOOP_BLOCKED: <one-line reason> and stop.`;
  let iter = 0;
  let lastAudit = "";
  let sameAuditStreak = 0;
  let anyMutationSinceAudit = false;

  try {
    while (true) {
      iter++;
      if (maxIter > 0 && iter > maxIter) {
        status(`stopped: --max ${maxIter} iterations reached (goal not verified)`);
        showChanges(loopStart, "total");
        process.exit(2);
      }
      const iterStart = cp.available() ? cp.head() : null;
      status(`loop ${iter} — working`);
      const work = await turn(nextMsg);
      if (work.mutated) { anyMutationSinceAudit = true; checkpoint(`loop ${iter}: work`); }

      const blocked = work.text.match(/LOOP_BLOCKED:?\s*(.*)/);
      if (blocked && !work.mutated) {
        status(`stopped: goal not applicable — ${blocked[1]?.trim() || "see output above"}`);
        showChanges(loopStart, "total");
        process.exit(3);
      }

      // Gate 1: objective checks — zero model tokens, hard evidence.
      const failed = runChecks();
      if (failed) {
        showChanges(iterStart, `loop ${iter}`);
        status(`loop ${iter} — check failed: ${failed.cmd}`);
        nextMsg = `[loop] Verification command failed:\n$ ${failed.cmd}\n${failed.out}\n\nFix the failure, then continue toward the goal.`;
        continue;
      }

      // Gate 2: model audit against the original goal. An audit that ran
      // mutating tools (often just bash used for read-only greps) gets a
      // confirmation pass instead of a wasted full work iteration; after
      // 3 consecutive LOOP_DONE audits we trust it regardless.
      status(`loop ${iter} — verifying`);
      let audit = await turn(VERIFY_PROMPT);
      let doneVotes = /\bLOOP_DONE\b/.test(audit.text) ? 1 : 0;
      while (doneVotes >= 1 && doneVotes < 3 && audit.mutated) {
        anyMutationSinceAudit = true;
        checkpoint(`loop ${iter}: verify pass ${doneVotes}`);
        status(`loop ${iter} — audit used tools, confirming`);
        audit = await turn(VERIFY_PROMPT);
        if (/\bLOOP_DONE\b/.test(audit.text)) doneVotes++;
        else doneVotes = 0;
      }
      if (audit.mutated) { anyMutationSinceAudit = true; checkpoint(`loop ${iter}: verify fixes`); }

      showChanges(iterStart, `loop ${iter}`);

      const auditBlocked = audit.text.match(/LOOP_BLOCKED:?\s*(.*)/);
      if (auditBlocked && !audit.mutated) {
        status(`stopped: goal not applicable — ${auditBlocked[1]?.trim() || "see output above"}`);
        showChanges(loopStart, "total");
        process.exit(3);
      }

      if (doneVotes >= 1 && (!audit.mutated || doneVotes >= 3)) {
        // Belt-and-braces: audit passed AND checks pass on the final state.
        const recheck = runChecks();
        if (!recheck) {
          status(`done after ${iter} iteration${iter === 1 ? "" : "s"} — goal verified`);
          showChanges(loopStart, "total");
          process.exit(0);
        }
        nextMsg = `[loop] Final check failed after your LOOP_DONE:\n$ ${recheck.cmd}\n${recheck.out}\n\nFix it.`;
        continue;
      }

      // Stall detection: identical audit twice with no mutations in between.
      const sig = norm(audit.text);
      if (sig && sig === lastAudit && !anyMutationSinceAudit) sameAuditStreak++;
      else sameAuditStreak = 0;
      lastAudit = sig;
      anyMutationSinceAudit = false;
      if (sameAuditStreak >= 2) {
        status(`stopped: stalled — the same unmet items repeated ${sameAuditStreak + 1}× with no progress. Remaining:\n${audit.text.trim()}`);
        showChanges(loopStart, "total");
        process.exit(2);
      }

      nextMsg = CONTINUE_PROMPT;

      // Compaction: summarize into a fresh session before context blows up.
      if (historySize() > budget) {
        status(`loop ${iter} — compacting context`);
        const summary = await turn(COMPACT_PROMPT);
        session = Session.create(config.dataDir, { cwd: config.cwd, model: provider.model, at: new Date().toISOString() });
        config.sessionId = session.id;
        nextMsg = `[loop mode] GOAL:\n${goal}\n\nProgress handoff from the previous context:\n${summary.text.trim()}\n\nContinue until the goal is FULLY complete; an auditor re-verifies everything.`;
      }
    }
  } catch (e) {
    if (ctrl.signal.aborted) {
      status(`aborted at iteration ${iter}`);
      showChanges(loopStart, "total");
      process.exit(130);
    }
    if (json) process.stdout.write(JSON.stringify({ type: "error", message: (e as Error).message }) + "\n");
    else console.error(`\nloop failed: ${(e as Error).message}`);
    process.exit(1);
  }
}
