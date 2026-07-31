import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { resolvePath } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

export async function writeTool(
  args: { path: string; content: string },
  ctx: ToolCtx,
): Promise<string> {
  const path = resolvePath(args.path, ctx.cwd);
  mkdirSync(dirname(path), { recursive: true });
  const n = await Bun.write(path, args.content);
  return `wrote ${n} bytes to ${path}`;
}
