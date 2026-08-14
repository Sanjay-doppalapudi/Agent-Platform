import { ensureReadable, isEnvFile, isSecretPath, looksBinaryByExt, redactEnvContent, resolvePath, sniffBinary, ToolError, canonicalPath } from "./shared.ts";
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
  // Cap the read before buffering the whole file into memory.
  const size = file.size;
  const cap = MAX_BYTES * 4;
  const bytes = new Uint8Array(await (size > cap ? file.slice(0, cap) : file).arrayBuffer());
  if (sniffBinary(bytes)) {
    return `binary file (${size || bytes.length} bytes): ${path}`;
  }

  let content = new TextDecoder().decode(bytes);
  if (ctx.config.redactEnv && (isEnvFile(path) || isSecretPath(path))) {
    if (isEnvFile(path) || isEnvFile(canonicalPath(path))) {
      content = redactEnvContent(content);
    } else {
      // JSON/YAML manifests and raw key material use different syntaxes. Do not
      // pretend the .env redactor covers them — suppress their body until a
      // format-aware redactor exists.
      return `secret file (${bytes.length} bytes): ${path} — contents redacted`;
    }
  }

  const lines = content.split("\n");
  // A trailing newline terminates the last line, it does not start a new one —
  // keeping it printed a phantom blank line past every file's end and made the
  // "more lines" hint over-count by one.
  if (lines.at(-1) === "") lines.pop();
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
