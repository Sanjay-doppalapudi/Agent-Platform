import { statSync } from "node:fs";
import { join } from "node:path";
import { allIgnores, ensureReadable, isPrivatePath, isIgnoredPath, privateExcludeGlobs, resolvePath } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const MAX_RESULTS = 300;
const MAX_SCAN = 10_000;

/**
 * File listing via `rg --files -g <pattern>` — ripgrep prunes ignored dirs
 * during traversal (Bun.Glob would walk node_modules/builds first and filter
 * later, which is unusable on the target repo). Falls back to Bun.Glob when
 * rg is missing.
 */
export async function globTool(
  args: { pattern: string; cwd?: string },
  ctx: ToolCtx,
): Promise<string> {
  const root = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  if (args.cwd) await ensureReadable(root, ctx);
  const ignores = allIgnores(ctx.config);

  let rels: string[];
  if (Bun.which("rg")) {
    const rgArgs = ["--files", "--hidden", "-g", args.pattern];
    for (const ig of ignores) rgArgs.push("-g", `!**/${ig}/**`, "-g", `!${ig}/**`);
    // Never enumerate AP-private data (read hard-denies it — listing it here
    // leaked transcript and credential filenames).
    if (ctx.config.sandbox !== "off") rgArgs.push(...privateExcludeGlobs(ctx.config, root));
    const proc = Bun.spawn(["rg", ...rgArgs], { cwd: root, stdout: "pipe", stderr: "ignore" });
    const out = await new Response(proc.stdout).text();
    await proc.exited;
    rels = out.split("\n").filter(Boolean).slice(0, MAX_SCAN);
  } else {
    const glob = new Bun.Glob(args.pattern);
    rels = [];
    for await (const rel of glob.scan({ cwd: root, dot: false, onlyFiles: true })) {
      if (isIgnoredPath(rel, ignores)) continue;
      if (ctx.config.sandbox !== "off" && isPrivatePath(join(root, rel), ctx.config)) continue;
      rels.push(rel);
      if (rels.length >= MAX_SCAN) break;
    }
  }

  const withMtime = rels.map((rel) => {
    let mtime = 0;
    try { mtime = statSync(join(root, rel)).mtimeMs; } catch {}
    return { rel, mtime };
  });
  withMtime.sort((a, b) => b.mtime - a.mtime);

  const shown = withMtime.slice(0, MAX_RESULTS);
  let out = shown.map((r) => r.rel).join("\n");
  const hidden = withMtime.length - shown.length;
  if (hidden > 0) out += `\n[+${hidden} more — narrow the pattern]`;
  return out || "(no matches)";
}
