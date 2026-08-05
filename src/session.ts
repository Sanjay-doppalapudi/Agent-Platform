// Append-only JSONL session store. Crash-safe by construction: a partial
// trailing line is ignored on load.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import type { Msg } from "./provider.ts";

export interface SessionMeta {
  cwd: string;
  model: string;
  at: string;
  /** Shadow-git checkpoint repo this session continues (set by /compact) so
   *  the undo trail survives compaction and restarts. */
  checkpointId?: string;
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
    this.history.push(msg);
    // A torn tail must be terminated in the SAME write, otherwise this record
    // concatenates onto the fragment and both become one unparsable line —
    // silently deleting this message from every future load.
    appendFileSync(this.file, (this.torn ? "\n" : "") + JSON.stringify({ t: "msg", ...msg }) + "\n");
    this.torn = false;
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
}
