import { ensureAllowed, resolvePath, ToolError } from "./shared.ts";
import type { ToolCtx } from "./index.ts";

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export async function editTool(
  rawArgs: { path: string; old?: string; new?: string; all?: boolean } & Record<string, unknown>,
  ctx: ToolCtx,
): Promise<string> {
  // Alias tolerance: weaker models call this with Claude-style names.
  const args = {
    path: rawArgs.path,
    old: (rawArgs.old ?? rawArgs.old_string ?? rawArgs.oldText ?? "") as string,
    new: (rawArgs.new ?? rawArgs.new_string ?? rawArgs.newText ?? "") as string,
    all: Boolean(rawArgs.all ?? rawArgs.replace_all),
  };
  const path = resolvePath(args.path, ctx.cwd);
  await ensureAllowed(path, ctx, "edit file");
  const file = Bun.file(path);
  if (!(await file.exists())) throw new ToolError(`file not found: ${path}`);
  const content = await file.text();

  if (!args.old) {
    throw new ToolError(`edit requires {path, old, new} — put the exact text to replace in 'old' and the replacement in 'new'`);
  }

  const fileIsCRLF = content.includes("\r\n");
  let note = "";

  // Pass 1: exact match.
  let old = args.old;
  let nw = args.new;
  let count = content.split(old).length - 1;

  // Pass 2: model sent LF-only text for a CRLF file — the top Windows failure.
  if (count === 0 && fileIsCRLF && old.includes("\n") && !old.includes("\r\n")) {
    const oldCRLF = old.replace(/\n/g, "\r\n");
    const c = content.split(oldCRLF).length - 1;
    if (c > 0) {
      old = oldCRLF;
      nw = nw.replace(/(?<!\r)\n/g, "\r\n");
      count = c;
      note = " (matched after CRLF normalization)";
    }
  }

  if (count > 0) {
    if (count > 1 && !args.all) {
      throw new ToolError(`found ${count} matches in ${path} — add surrounding context to old, or pass all:true`);
    }
    // The replacement is arbitrary source code, NOT a regex substitution
    // template: a bare string here makes String.replace interpret $$, $&, $`
    // and $' — `$'` splices the whole file tail in, silently duplicating it.
    // A function replacer returns the text literally (same idiom as pass 3).
    const updated = args.all ? content.split(old).join(nw) : content.replace(old, () => nw);
    await Bun.write(path, updated);
    return `replaced ${args.all ? count : 1} occurrence${count > 1 && args.all ? "s" : ""} in ${path}${note}`;
  }

  // Pass 3: trailing-whitespace-insensitive line matching.
  if (args.old.includes("\n")) {
    const pattern = args.old
      .split(/\r?\n/)
      .map(escapeRe)
      .join("[ \\t]*\\r?\\n");
    const c = [...content.matchAll(new RegExp(pattern, "g"))].length;
    if (c > 0) {
      if (c > 1 && !args.all) {
        throw new ToolError(`found ${c} matches in ${path} — add surrounding context to old, or pass all:true`);
      }
      const replacement = fileIsCRLF ? args.new.replace(/(?<!\r)\n/g, "\r\n") : args.new;
      const re = new RegExp(pattern, args.all ? "g" : "");
      const updated = content.replace(re, () => replacement);
      await Bun.write(path, updated);
      return `replaced ${args.all ? c : 1} occurrence${c > 1 && args.all ? "s" : ""} in ${path} (matched ignoring trailing whitespace)`;
    }
  }

  throw new ToolError(
    `old string not found in ${path} (checked exact, CRLF, and trailing-whitespace variants) — read the file and copy the text exactly`,
  );
}
