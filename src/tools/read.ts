import { ensureReadable, isEnvFile, looksBinaryByExt, redactEnvContent, resolvePath, sniffBinary, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_LINES = 2000;
const MAX_BYTES = 50_000;
const MAX_LINE_CHARS = 500;

export async function readTool(
  args: { path: string; offset?: number; limit?: number },
  ctx: ToolCtx,
): Promise<string> {
  const path = resolvePath(args.path, ctx.cwd);
  await ensureReadable(path, ctx);
  const file = Bun.file(path);
  if (!(await file.exists())) throw new ToolError(`file not found: ${path}`);

  if (looksBinaryByExt(path)) {
    return `binary file (${file.size} bytes): ${path}`;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (sniffBinary(bytes)) {
    return `binary file (${bytes.length} bytes): ${path}`;
  }

  let content = new TextDecoder().decode(bytes);
  if (isEnvFile(path) && ctx.config.redactEnv) {
    content = redactEnvContent(content);
  }

  const lines = content.split("\n");
  const offset = Math.max(1, args.offset ?? 1);
  const limit = Math.min(args.limit ?? MAX_LINES, MAX_LINES);
  const slice = lines.slice(offset - 1, offset - 1 + limit);

  let out = "";
  let budget = MAX_BYTES;
  let shown = 0;
  for (let i = 0; i < slice.length; i++) {
    let line = slice[i]!;
    if (line.length > MAX_LINE_CHARS) line = line.slice(0, MAX_LINE_CHARS) + "…";
    const numbered = `${offset + i}: ${line}\n`;
    budget -= numbered.length;
    if (budget < 0) break;
    out += numbered;
    shown++;
  }
  const end = offset - 1 + shown;
  if (end < lines.length) out += `[${lines.length - end} more lines — use offset=${end + 1}]`;
  return out || "(empty file)";
}
