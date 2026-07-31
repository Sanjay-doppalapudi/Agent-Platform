import { resolvePath, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

export async function editTool(
  args: { path: string; old: string; new: string; all?: boolean },
  ctx: ToolCtx,
): Promise<string> {
  const path = resolvePath(args.path, ctx.cwd);
  const file = Bun.file(path);
  if (!(await file.exists())) throw new ToolError(`file not found: ${path}`);
  const content = await file.text();

  if (!args.old) throw new ToolError("old must be non-empty");
  const count = content.split(args.old).length - 1;
  if (count === 0) {
    throw new ToolError(`old string not found in ${path} — read the file and match exactly (whitespace matters)`);
  }
  if (count > 1 && !args.all) {
    throw new ToolError(`found ${count} matches in ${path} — add surrounding context to old, or pass all:true`);
  }

  const updated = args.all
    ? content.split(args.old).join(args.new)
    : content.replace(args.old, args.new);
  await Bun.write(path, updated);
  return `replaced ${args.all ? count : 1} occurrence${count > 1 && args.all ? "s" : ""} in ${path}`;
}
