// Append-only JSONL session store. Crash-safe by construction: a partial
// trailing line is ignored on load.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import type { Msg } from "./provider.ts";

export interface SessionMeta {
  cwd: string;
  model: string;
  at: string;
}

export class Session {
  history: Msg[] = [];
  constructor(
    public id: string,
    private file: string,
  ) {}

  append(msg: Msg) {
    this.history.push(msg);
    appendFileSync(this.file, JSON.stringify({ t: "msg", ...msg }) + "\n");
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
    for (const line of readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.t === "msg") {
          const { t, ...msg } = obj;
          s.history.push(msg as Msg);
        }
      } catch {} // partial trailing line — ignore
    }
    return s;
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
