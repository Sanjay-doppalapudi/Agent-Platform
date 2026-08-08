// Shared compaction: summarize → archive index → fresh session with
// parentSessionId. Optional auto-memory synthesis into the repo-keyed dir.
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fireLifecycle } from "./agent.ts";
import type { Config, ResolvedProvider } from "./config.ts";
import { ensureRepoMemoryDir, readMemories } from "./memory.ts";
import { streamChat } from "./provider.ts";
import { Session } from "./session.ts";

export interface ArchiveEntry {
  at: string;
  oldId: string;
  newId: string;
  summaryChars: number;
  reason: string;
  cwd: string;
  title?: string;
}

export function archivesDir(dataDir: string): string {
  const d = join(dataDir, "archives");
  mkdirSync(d, { recursive: true });
  return d;
}

export function appendArchive(dataDir: string, entry: ArchiveEntry) {
  const file = join(archivesDir(dataDir), "index.jsonl");
  appendFileSync(file, JSON.stringify(entry) + "\n");
}

export function listArchives(dataDir: string, limit = 20): ArchiveEntry[] {
  const file = join(archivesDir(dataDir), "index.jsonl");
  if (!existsSync(file)) return [];
  const out: ArchiveEntry[] = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line) as ArchiveEntry);
    } catch {
      // torn / corrupt — skip
    }
  }
  return out.slice(-limit).reverse();
}

const SUMMARIZE_SYSTEM =
  "Summarize this coding session for a successor agent: goals, decisions, files touched, current state, open items. Be concise but complete. Output only the summary.";

const MEMORY_EXTRACT_SYSTEM =
  `From the session summary, extract 0 to 3 durable preference/technique memories. Each memory is exactly three lines:
Title: …
User wanted: …
Why (guess): …
Separate memories with a line containing only ---. If nothing is worth remembering, reply with NONE.`;

export interface CompactResult {
  oldId: string;
  session: Session;
  summary: string;
  memoriesWritten: number;
}

/**
 * REPL-style compact: no-tools summary via streamChat, then fresh session.
 */
export async function compactSession(opts: {
  session: Session;
  config: Config;
  provider: ResolvedProvider;
  checkpointId: string;
  reason: "manual" | "auto" | "loop";
  idleTimeoutMs?: number;
  /** When set (loop path), skip streamChat and use this summary text. */
  summaryText?: string;
  seedMode?: "repl" | "loop";
  loopGoal?: string;
  loopPending?: string;
}): Promise<CompactResult> {
  const { session, config, provider, checkpointId, reason } = opts;
  const oldId = session.id;

  fireLifecycle(config, "preCompact", {
    sessionId: oldId,
    cwd: config.cwd,
    reason,
  });

  let summary = (opts.summaryText ?? "").trim();
  if (!summary) {
    const msgs = [
      { role: "system" as const, content: SUMMARIZE_SYSTEM },
      ...session.history,
      { role: "user" as const, content: "Summarize the session now." },
    ];
    await streamChat(
      provider,
      msgs,
      [],
      (d) => { summary += d; },
      undefined,
      undefined,
      undefined,
      opts.idleTimeoutMs ?? config.streamIdleSeconds * 1000,
    );
    summary = summary.trim();
  }

  const next = Session.create(config.dataDir, {
    cwd: config.cwd,
    model: provider.model,
    at: new Date().toISOString(),
    checkpointId,
    parentSessionId: oldId,
    compactReason: reason,
    ...(session.meta?.title ? { title: session.meta.title } : {}),
  });

  appendArchive(config.dataDir, {
    at: new Date().toISOString(),
    oldId,
    newId: next.id,
    summaryChars: summary.length,
    reason,
    cwd: config.cwd,
    title: session.meta?.title,
  });

  if (opts.seedMode === "loop") {
    // Loop folds the next work message into the handoff; caller appends it.
  } else {
    next.append({ role: "user", content: `[Compacted context from session ${oldId}]\n${summary}` });
    next.append({ role: "assistant", content: "Context loaded — continuing from the summary." });
  }

  let memoriesWritten = 0;
  if (!config.light && config.autoMemory !== "off" && summary) {
    memoriesWritten = await synthesizeMemories(config, provider, summary, opts.idleTimeoutMs);
  }

  fireLifecycle(config, "postCompact", {
    sessionId: next.id,
    parentSessionId: oldId,
    cwd: config.cwd,
    summaryChars: summary.length,
    reason,
    memoriesWritten,
  });

  return { oldId, session: next, summary, memoriesWritten };
}

/** Build the loop's next user message after compact. */
export function loopCompactSeed(goal: string, summary: string, pending: string): string {
  return `[loop mode] GOAL:\n${goal}\n\nProgress handoff from the previous context:\n${summary.trim()}\n\n${pending}`;
}

async function synthesizeMemories(
  config: Config,
  provider: ResolvedProvider,
  summary: string,
  idleTimeoutMs?: number,
): Promise<number> {
  let raw = "";
  try {
    await streamChat(
      provider,
      [
        { role: "system", content: MEMORY_EXTRACT_SYSTEM },
        { role: "user", content: summary.slice(0, 12_000) },
      ],
      [],
      (d) => { raw += d; },
      undefined,
      undefined,
      undefined,
      idleTimeoutMs ?? config.streamIdleSeconds * 1000,
    );
  } catch {
    return 0;
  }
  const cards = parseMemoryCards(raw);
  if (!cards.length) return 0;
  const dir = ensureRepoMemoryDir(config.dataDir, config.cwd);
  const existing = readMemories(dir).toLowerCase();
  let written = 0;
  for (const card of cards.slice(0, 3)) {
    const norm = card.toLowerCase().replace(/\s+/g, " ");
    if (existing.includes(norm.slice(0, 80))) continue;
    const titleLine = card.split("\n").find((l) => /^title:/i.test(l)) ?? "Title: note";
    const slug = slugify(titleLine.replace(/^title:\s*/i, "")) || `mem-${written + 1}`;
    const path = join(dir, `${slug}.md`);
    if (existsSync(path)) continue;
    try {
      writeFileSync(path, card.trim() + "\n");
      written++;
    } catch {}
  }
  return written;
}

/** Parse model memory extraction output into card bodies. */
export function parseMemoryCards(raw: string): string[] {
  const t = raw.trim();
  if (!t || /^none\b/i.test(t)) return [];
  const parts = t.split(/\n---\n/).map((p) => p.trim()).filter(Boolean);
  const cards: string[] = [];
  for (const p of parts) {
    if (!/^title:/im.test(p) || !/user wanted:/im.test(p)) continue;
    const lines = p.split("\n").map((l) => l.trim()).filter(Boolean);
    const title = lines.find((l) => /^title:/i.test(l));
    const wanted = lines.find((l) => /^user wanted:/i.test(l));
    const why = lines.find((l) => /^why \(guess\):/i.test(l)) ?? "Why (guess): (unspecified)";
    if (title && wanted) cards.push([title, wanted, why].join("\n"));
  }
  return cards;
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

/**
 * Load a note from an archived parent session for /restore-context.
 * Returns a user-message body, or null if unavailable.
 */
export function restoreContextNote(dataDir: string, sessionId: string): string | null {
  try {
    const s = Session.load(dataDir, sessionId);
    const summaryMsg = s.history.find(
      (m) => m.role === "user" && typeof m.content === "string" && m.content.startsWith("[Compacted context"),
    );
    if (summaryMsg && typeof summaryMsg.content === "string") {
      return `[Restored context from session ${sessionId}]\n${summaryMsg.content.replace(/^\[Compacted context from session [^\]]+\]\n/, "")}`;
    }
    // Fall back: last user + assistant pair
    let lastUser = "";
    let lastAsst = "";
    for (const m of s.history) {
      if (m.role === "user" && typeof m.content === "string") lastUser = m.content;
      if (m.role === "assistant" && typeof m.content === "string") lastAsst = m.content;
    }
    if (!lastUser && !lastAsst) return null;
    return `[Restored context from session ${sessionId}]\nUser: ${lastUser.slice(0, 4000)}\nAssistant: ${lastAsst.slice(0, 4000)}`;
  } catch {
    return null;
  }
}

