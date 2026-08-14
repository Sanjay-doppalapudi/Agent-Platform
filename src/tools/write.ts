import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { ensureAllowed, isInsideRoots, resolvePath, ToolError } from "./shared.ts";
import { isValidMemoryCard, normalizeMemoryCard } from "../memory.ts";
import type { ToolCtx } from "./index.ts";

export async function writeTool(
  args: { path: string; content: string },
  ctx: ToolCtx,
): Promise<string> {
  const path = resolvePath(args.path, ctx.cwd);
  if (typeof args.content !== "string") {
    throw new ToolError("write requires {path, content}");
  }
  await ensureAllowed(path, ctx, "write file");
  // Memory cards are injected into the system prompt — force the three-line
  // schema so free-form prompt-injection text cannot persist across sessions.
  const memRoot = join(ctx.config.dataDir, "memory");
  if (isInsideRoots(path, [memRoot]) && path.toLowerCase().endsWith(".md")) {
    const norm = normalizeMemoryCard(args.content);
    if (!norm || !isValidMemoryCard(args.content)) {
      throw new ToolError(
        "memory cards must be exactly three lines: \"Title: …\", \"User wanted: …\", \"Why (guess): …\" (no free-form Markdown)",
      );
    }
    mkdirSync(dirname(path), { recursive: true });
    const n = await Bun.write(path, norm);
    return `wrote ${n} bytes to ${path}`;
  }
  mkdirSync(dirname(path), { recursive: true });
  const n = await Bun.write(path, args.content);
  return `wrote ${n} bytes to ${path}`;
}
