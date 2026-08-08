// In-process agent channels — append-only JSONL under <dataDir>/channels/.
// Posts drain into the NEXT user message (same contract as tasks/steer).
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export interface ChannelPost {
  at: string;
  from: string;
  to?: string;
  text: string;
}

const pending = new Map<string, string[]>();

function channelFile(dataDir: string, id: string): string {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64) || "default";
  const dir = join(dataDir, "channels");
  mkdirSync(dir, { recursive: true });
  return join(dir, `${safe}.jsonl`);
}

/** Validate channel id (user/model controlled). */
export function safeChannelId(id: string): string | null {
  const t = id.trim();
  if (!t || t.length > 64 || !/^[\w-]+$/.test(t)) return null;
  return t;
}

export function postChannel(
  dataDir: string,
  channelId: string,
  from: string,
  text: string,
  to?: string,
): boolean {
  const id = safeChannelId(channelId);
  if (!id) return false;
  const body = text.replace(/\s+/g, " ").trim().slice(0, 4_000);
  if (!body) return false;
  const entry: ChannelPost = {
    at: new Date().toISOString(),
    from: from.slice(0, 80),
    text: body,
    ...(to ? { to: to.slice(0, 80) } : {}),
  };
  appendFileSync(channelFile(dataDir, id), JSON.stringify(entry) + "\n");
  const note = `<channel name="${id}" from="${entry.from}">${body}</channel>`;
  const list = pending.get(id) ?? [];
  list.push(note);
  pending.set(id, list);
  return true;
}

/** Destructive drain of in-memory notes since last turn (all channels). */
export function drainChannelNotes(): string[] {
  const out: string[] = [];
  for (const [, notes] of pending) out.push(...notes);
  pending.clear();
  return out;
}

export function pendingChannelCount(): number {
  let n = 0;
  for (const notes of pending.values()) n += notes.length;
  return n;
}

/** Read recent posts from disk (non-destructive). */
export function readChannel(dataDir: string, channelId: string, limit = 20): ChannelPost[] {
  const id = safeChannelId(channelId);
  if (!id) return [];
  const file = channelFile(dataDir, id);
  if (!existsSync(file)) return [];
  const rows: ChannelPost[] = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line) as ChannelPost); } catch {}
  }
  return rows.slice(-limit);
}
