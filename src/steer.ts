// Mid-turn steering queue. Notes fold into the NEXT user message (same
// contract as tasks.ts drainTaskNotes) — never injected mid-turn, so
// history stays append-only and prefix caching stays intact.
const notes: string[] = [];

/** Queue a coaching note for the next turn. Empty/whitespace is ignored. */
export function pushSteer(text: string): boolean {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t) return false;
  notes.push(t.slice(0, 4_000));
  return true;
}

/** Destructive drain — each note reaches the model exactly once. */
export function drainSteerNotes(): string[] {
  return notes.splice(0, notes.length);
}

export function pendingSteerCount(): number {
  return notes.length;
}
