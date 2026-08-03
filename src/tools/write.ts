import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { ensureAllowed, resolvePath, ToolError } from "./shared.ts";
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
  mkdirSync(dirname(path), { recursive: true });
  const n = await Bun.write(path, args.content);
  return `wrote ${n} bytes to ${path}`;
}
