import { allIgnores, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_BYTES = 30_000;
const MAX_MATCHES = 200;

export async function grepTool(
  args: {
    pattern: string;
    path?: string;
    glob?: string;
    mode?: "files" | "content" | "count";
    ignoreCase?: boolean;
    context?: number;
  },
  ctx: ToolCtx,
): Promise<string> {
  if (!Bun.which("rg")) {
    throw new ToolError("ripgrep (rg) not found on PATH — install from https://github.com/BurntSushi/ripgrep");
  }
  const root = args.path ? resolvePath(args.path, ctx.cwd) : ctx.cwd;
  const mode = args.mode ?? "content";

  const rgArgs = ["--no-messages", "--hidden"];
  for (const ig of allIgnores(ctx.config)) rgArgs.push("-g", `!**/${ig}/**`, "-g", `!${ig}/**`);
  if (args.glob) rgArgs.push("-g", args.glob);
  if (args.ignoreCase) rgArgs.push("-i");
  switch (mode) {
    case "files": rgArgs.push("-l"); break;
    case "count": rgArgs.push("--count-matches"); break;
    default:
      rgArgs.push("-n", "-m", String(MAX_MATCHES));
      if (args.context) rgArgs.push("-C", String(Math.min(args.context, 10)));
  }
  rgArgs.push("-e", args.pattern, ".");

  const proc = Bun.spawn(["rg", ...rgArgs], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const [out, err, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  if (code !== 0 && code !== 1) {
    throw new ToolError(`rg failed (${code}): ${err.slice(0, 300)}`);
  }
  if (!out.trim()) return "(no matches)";

  let lines = out.split("\n").filter(Boolean);
  let note = "";
  if (lines.length > MAX_MATCHES) {
    note = `\n[showing ${MAX_MATCHES} of ${lines.length} lines — narrow the pattern]`;
    lines = lines.slice(0, MAX_MATCHES);
  }
  return truncateMiddle(lines.join("\n"), MAX_BYTES) + note;
}
