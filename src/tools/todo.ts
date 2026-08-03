// todo tool: per-session in-memory checklist so long tasks stay on rails.
// State lives in the process (REPL/server lifetime) — deliberately ephemeral.
import type { ToolCtx } from "./index.ts";

interface TodoItem { text: string; done: boolean }
const lists = new Map<string, TodoItem[]>();

function render(items: TodoItem[]): string {
  if (!items.length) return "(todo list empty)";
  const open = items.filter((i) => !i.done).length;
  const lines = items.map((i, n) => `${i.done ? "[x]" : "[ ]"} ${n + 1}. ${i.text}`);
  return `${open} open · ${items.length - open} done\n${lines.join("\n")}`;
}

export async function todoTool(
  args: { add?: string[] | string; done?: number[] | number; clear?: boolean },
  ctx: ToolCtx,
): Promise<string> {
  const key = ctx.config.sessionId ?? "default";
  let items = lists.get(key);
  if (!items) { items = []; lists.set(key, items); }

  if (args.clear) items.length = 0;
  const adds = typeof args.add === "string" ? [args.add] : args.add ?? [];
  for (const t of adds) if (typeof t === "string" && t.trim()) items.push({ text: t.trim().slice(0, 200), done: false });
  const dones = typeof args.done === "number" ? [args.done] : args.done ?? [];
  for (const n of dones) {
    const item = items[Number(n) - 1];
    if (item) item.done = true;
  }
  if (items.length > 50) items.splice(0, items.length - 50);
  return render(items);
}
