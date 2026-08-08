// Append-only JSONL session store. Crash-safe by construction: a partial
// trailing line is ignored on load.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Msg } from "./provider.ts";

/**
 * Guarantee every stored tool_call carries a valid stringified JSON OBJECT.
 *
 * A model that emits malformed arguments (e.g. `{"path":"x", " "content":…`)
 * used to poison the session permanently: the tool itself failed cleanly, but
 * the raw string was persisted and re-sent on every later request, and
 * providers reject the entire conversation with HTTP 400 ("tool arguments
 * must be a stringified JSON object"). One bad call ended the session — and
 * survived /resume, because it was on disk.
 *
 * Repairs are attempted first (parseArgsLenient handles fences, smart quotes,
 * trailing commas, double-encoding); a hopeless value becomes "{}" so the
 * assistant/tool pairing stays intact and the tool-result message is what
 * tells the model its call was malformed. Well-formed calls are left byte-
 * identical. Returns true when anything was changed.
 */
export function sanitizeToolCallArgs(msg: Msg): boolean {
  const calls = (msg as any)?.tool_calls;
  if (!Array.isArray(calls)) return false;
  let changed = false;
  for (const tc of calls) {
    const fn = tc?.function;
    if (!fn) continue;
    const raw = fn.arguments;
    if (typeof raw === "string") {
      try {
        const v = JSON.parse(raw);
        if (v !== null && typeof v === "object" && !Array.isArray(v)) continue; // already a JSON object
      } catch {}
    }
    let repaired = "{}";
    try {
      // Lazy require: session.ts is on the startup path, tools/index.ts is not.
      const { parseArgsLenient } = require("./tools/index.ts") as typeof import("./tools/index.ts");
      const v = parseArgsLenient(typeof raw === "string" ? raw : JSON.stringify(raw ?? {}));
      if (v !== null && typeof v === "object" && !Array.isArray(v)) repaired = JSON.stringify(v);
    } catch {}
    fn.arguments = repaired;
    changed = true;
  }
  return changed;
}

export interface SessionMeta {
  cwd: string;
  model: string;
  at: string;
  /** Shadow-git checkpoint repo this session continues (set by /compact) so
   *  the undo trail survives compaction and restarts. */
  checkpointId?: string;
  /** Optional human label — does not rename the file (mtime/`--continue`
   *  still key on id). Appended as a later meta line; load() last-wins. */
  title?: string;
  /** Prior session id when this session was created by compaction. */
  parentSessionId?: string;
  /** Why compaction ran (manual | auto | loop). */
  compactReason?: string;
}

export class Session {
  history: Msg[] = [];
  meta: SessionMeta | null = null;
  /** A line failed to parse on load, or a torn tail was repaired. */
  recovered = false;
  /** The file does not end in a newline (crash mid-write). */
  private torn = false;

  constructor(
    public id: string,
    private file: string,
  ) {}

  append(msg: Msg) {
    sanitizeToolCallArgs(msg);
    this.history.push(msg);
    // A torn tail must be terminated in the SAME write, otherwise this record
    // concatenates onto the fragment and both become one unparsable line —
    // silently deleting this message from every future load.
    appendFileSync(this.file, (this.torn ? "\n" : "") + JSON.stringify({ t: "msg", ...msg }) + "\n");
    this.torn = false;
  }

  /** Append a meta line (load() last-wins). Used for title / checkpointId. */
  writeMeta(meta: SessionMeta) {
    appendFileSync(this.file, (this.torn ? "\n" : "") + JSON.stringify({ t: "meta", ...meta }) + "\n");
    this.torn = false;
    this.meta = meta;
  }

  static dir(dataDir: string): string {
    const d = join(dataDir, "sessions");
    mkdirSync(d, { recursive: true });
    return d;
  }

  static create(dataDir: string, meta: SessionMeta): Session {
    const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
    const id = `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
    const file = join(Session.dir(dataDir), `${id}.jsonl`);
    appendFileSync(file, JSON.stringify({ t: "meta", ...meta }) + "\n");
    return new Session(id, file);
  }

  static load(dataDir: string, id: string): Session {
    const file = join(Session.dir(dataDir), `${id}.jsonl`);
    if (!existsSync(file)) throw new Error(`session not found: ${id}`);
    const s = new Session(id, file);
    const raw = readFileSync(file, "utf8");
    // Repair lazily on the next append, never here: loads must not touch the
    // file, because list()/latest() rank by mtime and merely viewing an old
    // session would then hijack `ap --continue`.
    s.torn = raw.length > 0 && !raw.endsWith("\n");
    if (s.torn) s.recovered = true;
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.t === "msg") {
          const { t, ...msg } = obj;
          // Heal sessions written before the sanitizer existed: a stored
          // tool_call whose arguments are not valid JSON is re-sent verbatim
          // and the provider rejects the WHOLE request (HTTP 400
          // "tool arguments must be a stringified JSON object"), so such a
          // session could never take another turn. Repair in memory only —
          // load must not write (list()/latest() rank by mtime).
          if (sanitizeToolCallArgs(msg as Msg)) s.recovered = true;
          s.history.push(msg as Msg);
        } else if (obj.t === "meta") {
          const { t, ...meta } = obj;
          s.meta = meta as SessionMeta;
        }
      } catch {
        s.recovered = true; // partial/corrupt line — skipped, reported once
      }
    }
    return s;
  }

  static list(dataDir: string, limit = 10): { id: string; mtime: number }[] {
    const dir = Session.dir(dataDir);
    return readdirSync(dir)
      .filter((f) => f.endsWith(".jsonl"))
      .map((f) => ({ id: f.slice(0, -6), mtime: statSync(join(dir, f)).mtimeMs }))
      .sort((a, b) => b.mtime - a.mtime)
      .slice(0, limit);
  }

  static latest(dataDir: string): string | null {
    const dir = Session.dir(dataDir);
    let best: { id: string; mtime: number } | null = null;
    for (const f of readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      const mtime = statSync(join(dir, f)).mtimeMs;
      if (!best || mtime > best.mtime) best = { id: f.slice(0, -6), mtime };
    }
    return best?.id ?? null;
  }

  static delete(dataDir: string, id: string) {
    const file = join(Session.dir(dataDir), `${id}.jsonl`);
    if (existsSync(file)) unlinkSync(file);
  }

  /**
   * Drop the last `n` user turns and everything after each cut point's first
   * dropped user message. Rewrites the JSONL file atomically (temp → rename).
   * Does not touch the workspace — pair with /undo for file rollback.
   */
  rewind(n = 1): number {
    if (n < 1) return 0;
    let userCount = 0;
    let cut = this.history.length;
    for (let i = this.history.length - 1; i >= 0; i--) {
      if (this.history[i]!.role === "user") {
        userCount++;
        cut = i;
        if (userCount >= n) break;
      }
    }
    if (userCount === 0) return 0;
    const dropped = this.history.length - cut;
    this.history = this.history.slice(0, cut);
    const tmp = this.file + ".tmp";
    const lines: string[] = [];
    if (this.meta) lines.push(JSON.stringify({ t: "meta", ...this.meta }));
    for (const msg of this.history) {
      sanitizeToolCallArgs(msg);
      lines.push(JSON.stringify({ t: "msg", ...msg }));
    }
    writeFileSync(tmp, lines.length ? lines.join("\n") + "\n" : "");
    renameSync(tmp, this.file);
    this.torn = false;
    return dropped;
  }

  /** Set/clear the session title by appending a meta line (load last-wins).
   *  Empty title clears it. Returns the loaded session or throws if missing. */
  static rename(dataDir: string, id: string, title: string): Session {
    const s = Session.load(dataDir, id);
    const cleaned = title.replace(/\s+/g, " ").trim().slice(0, 120);
    const meta: SessionMeta = {
      cwd: s.meta?.cwd ?? "",
      model: s.meta?.model ?? "",
      at: s.meta?.at ?? new Date().toISOString(),
      checkpointId: s.meta?.checkpointId,
      parentSessionId: s.meta?.parentSessionId,
      compactReason: s.meta?.compactReason,
      ...(cleaned ? { title: cleaned } : {}),
    };
    s.writeMeta(meta);
    return s;
  }
}
