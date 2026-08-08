// Repo outline via ripgrep — definition-ish lines, not embeddings.
// Caps keep output small; never injected into the system prompt.
import { allIgnores, ensureReadable, privateExcludeGlobs, requireRg, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_FILES = 300;
const MAX_SYMBOLS = 200;
const MAX_BYTES = 40_000;

const DEF_PATTERN =
  "^\\s*(export\\s+)?(async\\s+)?(function|class|interface|type|enum|const|let|var|def |fn |pub\\s+(fn|struct|enum|trait)|func |package |struct |impl )";

export async function repomapTool(
  args: { path?: string; query?: string },
  ctx: ToolCtx,
): Promise<string> {
  const rg = requireRg();
  const root = args.path ? resolvePath(args.path, ctx.cwd) : ctx.cwd;
  if (args.path) await ensureReadable(root, ctx);

  const ignoreArgs: string[] = [];
  for (const ig of allIgnores(ctx.config)) ignoreArgs.push("-g", `!**/${ig}/**`, "-g", `!${ig}/**`);
  if (ctx.config.sandbox !== "off") ignoreArgs.push(...privateExcludeGlobs(ctx.config, root));

  // File inventory (newest-ish via rg --files; we sort by path for stability)
  const filesProc = Bun.spawn([rg, "--files", "--hidden", "--no-messages", ...ignoreArgs], {
    cwd: root, stdout: "pipe", stderr: "pipe",
  });
  const filesOut = await new Response(filesProc.stdout).text();
  await filesProc.exited;
  let files = filesOut.split("\n").filter(Boolean).sort();
  let fileNote = "";
  if (files.length > MAX_FILES) {
    fileNote = `\n[files truncated to ${MAX_FILES} of ${files.length}]`;
    files = files.slice(0, MAX_FILES);
  }

  const rgArgs = [
    "--no-messages", "--hidden", "-n",
    "-m", String(MAX_SYMBOLS * 2), // over-fetch then filter
    ...ignoreArgs,
    "-e", DEF_PATTERN,
    ".",
  ];

  const symProc = Bun.spawn([rg, ...rgArgs], { cwd: root, stdout: "pipe", stderr: "pipe" });
  const [symOut] = await Promise.all([
    new Response(symProc.stdout).text(),
    new Response(symProc.stderr).text(),
    symProc.exited,
  ]);

  const q = args.query?.trim().toLowerCase();
  let symLines = symOut.split("\n").filter(Boolean);
  if (q) symLines = symLines.filter((l) => l.toLowerCase().includes(q));
  let symNote = "";
  if (symLines.length > MAX_SYMBOLS) {
    symNote = `\n[symbols truncated to ${MAX_SYMBOLS}]`;
    symLines = symLines.slice(0, MAX_SYMBOLS);
  }

  const header = `Repo map under ${root}\n${files.length} files${fileNote}\n${symLines.length} symbol lines${symNote}\n`;
  const body = [
    "## Files",
    ...files.slice(0, 80).map((f) => `- ${f}`),
    files.length > 80 ? `… ${files.length - 80} more (use glob)` : "",
    "",
    "## Symbols",
    ...symLines,
  ].filter(Boolean).join("\n");

  return truncateMiddle(header + "\n" + body, MAX_BYTES) || "(empty map)";
}
