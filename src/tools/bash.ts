// Shell tool — SECURITY MODEL (full policy in SECURITY.md):
// Commands are authored by the model within a session the local user started
// and supervises; AP is a development tool, not a privilege boundary. Layered
// guardrails, all best-effort by design (a guardrail, not a VM):
//   1. scanDangerous — destructive patterns are BLOCKED outright (never
//      prompted) and logged for provider feedback
//   2. scanCmdPaths — path tokens outside the readable roots require an
//      interactive user permit; AP-private data (credentials, transcripts,
//      checkpoints) is hard-denied and cannot be permitted
//   3. timeouts with process-tree kill, output caps, ctrl+c abort,
//      background logs confined to the user's data dir with 7-day pruning
// Run genuinely untrusted code in a container/VM, not behind these checks.
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { canonicalPath, isInsideRoots, isPrivatePath, isRestrictedDataDirPath, readRoots, resolvePath, truncateMiddle, ToolError } from "./shared.ts";
import { isBlockedFetchHost, egressPolicyBlock } from "./fetch.ts";
import { agentChildEnv } from "../trust.ts";
import type { ToolCtx } from "./index.ts";
import { relative, isAbsolute } from "node:path";

// Dangerous-command patterns: auto-BLOCKED (never prompted), warned, and
// logged to <dataDir>/blocked-commands.jsonl so the user can report the model.
// Best-effort by design — obfuscated commands can evade a pattern scan.
const DANGEROUS: [RegExp, string][] = [
  [/\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\S*\s+("|')?(\/|[A-Za-z]:[\\/]|~\/?(\s|$)|\$HOME)/i, "recursive force delete of an absolute path"],
  [/\b(del|rmdir|rd)\b[^&|;]*\/s/i, "recursive Windows delete"],
  [/\bformat\s+[a-z]:/i, "disk format"],
  [/\breg(\.exe)?\s+(add|delete)\b/i, "registry modification"],
  [/remove-item\b[^&|;]*-recurse[^&|;]*([A-Za-z]:[\\/]|\\\\)/i, "recursive Remove-Item on an absolute path"],
  [/\bmkfs\b/, "filesystem format"],
  [/\bdd\s+[^&|;]*of=\/dev\//, "raw disk write"],
  // Command position only (start / after ;|&& / sudo) — mentioning the word
  // in an echo or commit message is not an attempt to run it.
  [/(^|[;&|]\s*|\bsudo\s+)shutdown\b/i, "system shutdown/restart"],
  [/\btaskkill\b[^&|;]*\/f[^&|;]*\/im/i, "force-kill processes by name"],
  [/:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;/, "fork bomb"],
  // Pipe-to-shell, including `curl … | sudo bash` (sudo was a prior bypass).
  [/\b(curl|wget|irm|iwr)\b[^&|;]*\|\s*(sudo\s+)?(bash|sh|iex|powershell)\b/i, "piping a download straight into a shell"],
  // Process substitution: bash <(curl …) downloads and executes without a pipe.
  [/\b(bash|sh)\s+<\(/i, "process-substitution shell of a download"],
  [/\b(bash|sh)\s+-c\s+["'][^"']*\b(curl|wget)\b/i, "shell -c wrapping a download"],
  // PowerShell download cradles (pipe form above misses `iex (iwr …)`).
  // Command-position / call-form only — mentioning the words in echo/commit text is fine.
  [/(^|[;&|]\s+)(iex|invoke-expression)\b/i, "PowerShell Invoke-Expression"],
  [/\biex\s*\(/i, "PowerShell Invoke-Expression"],
  [/\bpowershell\b[^&|;]*-(?:enc|encodedcommand)\b/i, "encoded PowerShell"],
  [/\bfind\b[^&|;]*-delete\b/i, "find -delete"],
  [/\b(chmod|chown|chgrp)\b[^&|;]*\s+\/(?:\s|$)/i, "permission change on filesystem root"],
  // Irreversible git ops — allow optional git global flags (-C path, -c key=val)
  // before the verb so `git -C . push --force` cannot bypass.
  [/\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*\s+push\b[^&|;]*--force\b|\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*\s+push\b[^&|;]*\s+-f(?:\s|$)/i, "git force-push"],
  [/\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*\s+reset\s+--hard\b/i, "git reset --hard"],
  [/\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*\s+clean\s+-[a-z]*f/i, "git clean -f"],
  [/\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*\s+remote\s+set-url\b/i, "git remote set-url"],
];

/**
 * Best-effort collapse so path-to-binary / eval / env prefixes don't dodge
 * pattern rails (`/usr/bin/git push`, `eval git push`, `git.exe push`).
 * Mirrors normalizeShellSegment in tools/index.ts (kept local to avoid a cycle).
 */
function collapseShellVerbs(cmd: string): string {
  let s = cmd;
  for (let i = 0; i < 8; i++) {
    const next = s
      // Bare `name.exe` with no directory prefix (git.exe, pwsh.exe) — the
      // natural Windows spelling. The path-basename branch below only strips
      // `.exe` when a path prefix is present, so `git.exe push --force` kept
      // its suffix and slipped past both `\bgit\b` rails AND the force-push
      // hard-block. Strip `.exe` from the token in any command position.
      .replace(/(^|[\s;&|(`]|\$\()([A-Za-z0-9_.+-]+?)\.exe\b/gi, "$1$2")
      .replace(/\b(?:command|builtin|exec|env|eval|source|nice|nohup|time|ionice)\s+/gi, "")
      .replace(/\bsudo(?:\s+-[nEn]*|\s+--\S+)*\s+/gi, "")
      // Path-to-binary → basename; allow spaces in dirs (Program Files).
      .replace(
        /(^|[\s;&|])(?:["'](((?:[A-Za-z]:)?[\\/]|\.\/|\.\.[\\/])[^"']+)["']|(((?:[A-Za-z]:)?[\\/]|\.\/|\.\.[\\/])(?:[^"'&|;]+[\\/])*[^\\/\s"']+?)(?:\.exe)?)(?=\s|$)/gi,
        (_m, lead, quoted, _qInner, bare) => {
          const p = String(quoted || bare || "").replace(/\.exe$/i, "");
          const base = p.split(/[\\/]/).pop() || p;
          return `${lead ?? ""}${base}`;
        },
      );
    if (next === s) break;
    s = next;
  }
  return s;
}

export function scanDangerous(cmd: string): string | null {
  for (const candidate of new Set([cmd, collapseShellVerbs(cmd)])) {
    for (const [re, why] of DANGEROUS) if (re.test(candidate)) return why;
  }
  return null;
}

/** Soft git rails: push / remote-add require a permit (or permission.bash allow). */
export function scanSensitiveGit(cmd: string): string | null {
  const gitFlags = String.raw`(?:\s+(?:-C\s+\S+|-c\s+\S+|--git-dir=\S+|--work-tree=\S+|-[^\s-]+))*`;
  const pushRe = new RegExp(String.raw`\bgit\b${gitFlags}\s+push\b`, "i");
  const remoteRe = new RegExp(String.raw`\bgit\b${gitFlags}\s+remote\s+add\b`, "i");
  for (const candidate of new Set([cmd, collapseShellVerbs(cmd)])) {
    if (pushRe.test(candidate)) return "git push";
    if (remoteRe.test(candidate)) return "git remote add";
  }
  return null;
}

// Absolute-ish path tokens inside a shell command (win drive, git-bash /c/,
// unix homes, ~, $HOME, %USERPROFILE%). Best-effort — this is the same
// guardrail-not-VM stance as scanDangerous.
//
// Windows drive letters use a lookbehind so `http://…` does NOT match as
// drive `P:` (the prior `p:/` false positive made URL commands look like
// in-workspace relative paths and skipped the outside-path gate).
//
// The bare `/` alternative must not fire mid-token. Matching every slash
// flagged `sed "s/foo/bar/g"`, `grep -r x src/` (→ the drive ROOT), `awk -F/`
// and even `1/2` as outside-the-workspace paths — which in a headless run
// (permits auto-deny) turned everyday commands into hard failures. A slash
// only begins a path token when nothing path-ish precedes it.
// A bare slash also needs a SECOND slash in the token to count as a path.
// Without that, Windows command switches (`dir /s`, `taskkill /f /im`,
// `robocopy … /e`, `findstr /i`) all resolved to C:\s, C:\f, C:\im … and were
// reported as outside-the-workspace paths. Real absolute paths worth gating
// (/etc/passwd, /home/other/.ssh/id_rsa, /workspace/leak, /c/Users/x) all have
// one. Trade-off: a bare `/tmp` with no child is not flagged.
// $HOMEDRIVE$HOMEPATH and the bare/braced $USERPROFILE bash forms address the
// Windows home dir too — a redirect to `"$USERPROFILE/.ap/trusted-workspaces.json"`
// used to extract NO token at all (empty scan → no gate), which let a bash
// redirect write the trust store. They are listed before $HOME so the longer
// $HOMEDRIVE… wins over a $HOME prefix match.
const PATH_TOKEN_RE = /(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?<![A-Za-z0-9._~/-])\/(?=[^\s"'`;|&<>()*]*\/)|~[\\/]|\$HOMEDRIVE\$HOMEPATH|\$\{?USERPROFILE\}?|\$HOME\b|%USERPROFILE%|\$env:USERPROFILE)[^\s"'`;|&<>()*]*/g;
const URL_TOKEN_RE = /\b[a-z][a-z0-9+.-]*:\/\/[^\s"'`;|&<>()*]+/gi;
/** Windows UNC / device paths: \\server\share, \\?\C:\…, \\.\pipe\… */
const UNC_TOKEN_RE = /\\\\[^\s"'`;|&<>]*/g;

/** Turn a file:// URL into a local filesystem path, or null if not file:. */
export function fileUrlToPath(raw: string): string | null {
  let u: URL;
  try { u = new URL(raw); } catch { return null; }
  if (u.protocol !== "file:") return null;
  // URL.pathname is percent-encoded and uses /; on Windows file:///C:/x → /C:/x
  let p = decodeURIComponent(u.pathname);
  if (process.platform === "win32" && /^\/[A-Za-z]:\//.test(p)) p = p.slice(1);
  // file://hostname/share (UNC) — treat as absolute UNC
  if (u.hostname && u.hostname !== "localhost" && u.hostname !== "127.0.0.1") {
    p = `\\\\${u.hostname}${p.replace(/\//g, "\\")}`;
  }
  return p.replace(/\//g, process.platform === "win32" ? "\\" : "/");
}

/** Hostnames/IPs inside a shell command that fetch would refuse (IMDS etc.). */
export function scanBlockedFetchUrls(cmd: string): string[] {
  const hits: string[] = [];
  for (const m of cmd.match(URL_TOKEN_RE) ?? []) {
    let host = "";
    try {
      const u = new URL(m);
      if (u.protocol !== "http:" && u.protocol !== "https:") continue;
      host = u.hostname;
    } catch { continue; }
    const why = isBlockedFetchHost(host);
    if (why) hits.push(`${m} (${why})`);
  }
  // Bare link-local / IMDS addresses (no scheme) and decimal/hex encodings.
  if (/\b169\.254\.\d{1,3}\.\d{1,3}\b/.test(cmd)) hits.push("169.254.x.x link-local address");
  if (/\bfd00:ec2::/i.test(cmd)) hits.push("fd00:ec2:: AWS IMDS");
  if (/\b0xa9fea9fe\b/i.test(cmd) || /\b2852039166\b/.test(cmd)) hits.push("encoded metadata IP");
  // curl --resolve / --connect-to can pin a public hostname to IMDS.
  if (/--(?:resolve|connect-to)\b[^&|;]*169\.254\./i.test(cmd)) {
    hits.push("curl --resolve/--connect-to to link-local");
  }
  // Followed redirects cannot be validated by the bash path scan — refuse -L.
  if (/\b(?:curl|wget)\b[^&|;]*\s(?:-L|--location)\b/i.test(cmd) && /\bhttps?:\/\//i.test(cmd)) {
    hits.push("curl/wget -L (redirects bypass host policy — use the fetch tool)");
  }
  // wget follows redirects by DEFAULT (unlike curl), so any http(s) wget is
  // an IMDS redirect risk unless max-redirect is forced to 0.
  if (/\bwget\b/i.test(cmd) && /\bhttps?:\/\//i.test(cmd) && !/--max-redirect\s*=?\s*0\b/i.test(cmd)) {
    hits.push("wget follows redirects by default — pass --max-redirect=0 or use the fetch tool");
  }
  return [...new Set(hits)].slice(0, 5);
}

/**
 * URL tokens in a command that `config.network` would refuse (deny / not in the
 * allowlist). Best-effort — a scanner cannot see a socket opened by a script,
 * so this is a speed bump, not the boundary; sandbox:"container" with
 * --network none is the real egress control. Metadata hosts are handled
 * separately (scanBlockedFetchUrls) and blocked under every policy.
 */
export function scanEgressPolicy(cmd: string, config: ToolCtx["config"]): string[] {
  const net = config.network;
  if (!net || net === "allow") return [];
  const hits: string[] = [];
  for (const m of cmd.match(URL_TOKEN_RE) ?? []) {
    try {
      const u = new URL(m);
      if (u.protocol !== "http:" && u.protocol !== "https:") continue;
      const why = egressPolicyBlock(net, u.hostname);
      if (why) hits.push(`${m} (${why})`);
    } catch { continue; }
  }
  return [...new Set(hits)].slice(0, 5);
}

let cachedRuntime: string | null | undefined;
/** docker or podman, whichever is on PATH. Cached; null when neither exists. */
export function containerRuntime(): string | null {
  if (cachedRuntime !== undefined) return cachedRuntime;
  cachedRuntime = Bun.which("docker") ?? Bun.which("podman") ?? null;
  return cachedRuntime;
}

/** Build the container argv for sandbox:"container": mount ONLY the workspace,
 *  egress off unless network:"allow", run the model's command as /bin/sh -c. */
export function containerArgv(config: ToolCtx["config"], cmd: string, cwd: string): string[] {
  const runtime = containerRuntime();
  if (!runtime) {
    throw new ToolError(
      "sandbox:\"container\" needs docker or podman on PATH, but neither was found — install one, or set sandbox to \"workspace\".",
    );
  }
  const net = config.network === "allow" ? [] : ["--network", "none"];
  const image = (typeof config.sandboxImage === "string" && config.sandboxImage.trim()) || "alpine";
  return [
    runtime, "run", "--rm", "-i",
    ...net,
    // Only the workspace is visible inside the container. dataDir (credentials,
    // sessions) lives elsewhere and is never mounted → structurally unreachable.
    "-v", `${cwd}:/workspace`,
    "-w", "/workspace",
    "--security-opt", "no-new-privileges",
    image, "/bin/sh", "-c", cmd,
  ];
}

/** Paths a command references outside the readable roots. {priv} = AP-private. */
export function scanCmdPaths(
  cmd: string,
  ctx: ToolCtx,
): { outside: string[]; priv: string[] } {
  const roots = readRoots(ctx.config);
  const outside = new Set<string>();
  const priv = new Set<string>();

  const consider = (raw: string) => {
    let p = raw
      .replace(/^~(?=[\\/])/, homedir())
      .replace(/^\$HOMEDRIVE\$HOMEPATH/, homedir())
      .replace(/^(\$HOME\b|%USERPROFILE%|\$env:USERPROFILE|\$\{?USERPROFILE\}?)/, homedir());
    p = resolvePath(p.replace(/[.,:]+$/, ""), ctx.cwd);
    const canonical = canonicalPath(p);
    if (isPrivatePath(p, ctx.config) || isRestrictedDataDirPath(p, ctx.config)) priv.add(p);
    if (isPrivatePath(canonical, ctx.config) || isRestrictedDataDirPath(canonical, ctx.config)) priv.add(canonical);
    if (!isInsideRoots(p, roots)) outside.add(p);
    if (!isInsideRoots(canonical, roots)) outside.add(canonical);
  };

  // file:// URLs are LOCAL paths — extract them BEFORE stripping other URLs,
  // otherwise `curl file:///…/credentials.json` slips past the private gate.
  for (const m of cmd.match(URL_TOKEN_RE) ?? []) {
    const fp = fileUrlToPath(m);
    if (fp) consider(fp);
  }
  // UNC / device paths (not matched by PATH_TOKEN_RE's drive-letter form).
  for (const m of cmd.match(UNC_TOKEN_RE) ?? []) consider(m);

  // Strip http(s) URLs before matching generic Unix absolute paths: otherwise
  // the `/path` suffix in `https://example.com/path` looks like a local file.
  // Keep file: out of the strip set — already handled above.
  const pathsOnly = cmd.replace(/\bhttps?:\/\/[^\s"'`;|&<>()*]+/gi, "");
  for (const m of pathsOnly.match(PATH_TOKEN_RE) ?? []) consider(m);

  // Relative escapes: ../ chains or a bare `cd ..` step out of the workspace.
  if (/(?:^|[\s"'=(])\.\.[\\/]|\bcd\s+\.\.(?![\w.])/.test(cmd)) outside.add("(relative path escaping the workspace via ..)");
  // When dataDir sits inside the workspace (./.ap), absolute-token scanning
  // never sees writes like `echo x > .ap/skills/evil`. Flag relative mentions
  // of non-writable dataDir subtrees the same way we flag private paths.
  const relData = relative(ctx.cwd, ctx.config.dataDir).replace(/\\/g, "/");
  if (relData && !relData.startsWith("..") && !isAbsolute(relData)) {
    const esc = relData.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`${esc}/(?!memory(?:/|$)|artifacts(?:/|$))[^\\s"'\\\`;|&<>]*`, "i");
    const hit = pathsOnly.replace(/\\/g, "/").match(re);
    if (hit) priv.add(resolvePath(hit[0]!, ctx.cwd));
  }
  // Prompt-poison paths under the workspace (.ap/skills, HARNESS.md, …).
  if (/(?:^|[\s"'=>(])(?:\.ap|\.claude)\/(?:skills|agents|workflows)\b/i.test(pathsOnly.replace(/\\/g, "/"))
    || /(?:^|[\s"'=>(])(?:AP|AGENTS|HARNESS|CLAUDE)\.md\b/i.test(pathsOnly)
    || /(?:^|[\s"'=>(])\.ap\/DECISIONS\.md\b/i.test(pathsOnly.replace(/\\/g, "/"))) {
    priv.add("(prompt-injected project instruction path)");
  }
  return { outside: [...outside].slice(0, 5), priv: [...priv].slice(0, 5) };
}

const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_TIMEOUT_MS = 600_000;
const MAX_OUTPUT_BYTES = 30_000;

let cachedShell: string[] | null = null;

/** Resolve the shell command prefix once per process. */
export function shellPrefix(pref: "auto" | "bash" | "powershell" | "cmd"): string[] {
  if (cachedShell && pref === "auto") return cachedShell;
  const gitBashCandidates = [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files\\Git\\usr\\bin\\bash.exe",
    "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
  ];
  const resolveBash = (): string | null => {
    for (const c of gitBashCandidates) if (existsSync(c)) return c;
    const w = Bun.which("bash");
    // System32 bash is the WSL shim — not usable as a project shell.
    if (w && !/system32/i.test(w)) return w;
    return null;
  };
  let out: string[];
  switch (pref) {
    case "bash": {
      const b = resolveBash();
      if (!b) throw new ToolError("shell=bash configured but Git Bash not found");
      out = [b, "-c"];
      break;
    }
    case "powershell":
      out = ["powershell", "-NoProfile", "-Command"];
      break;
    case "cmd":
      out = ["cmd", "/c"];
      break;
    default: {
      const b = process.platform === "win32" ? resolveBash() : Bun.which("bash");
      out = b ? [b, "-c"] : process.platform === "win32" ? ["powershell", "-NoProfile", "-Command"] : ["sh", "-c"];
    }
  }
  if (pref === "auto") cachedShell = out;
  return out;
}

function killTree(pid: number) {
  if (process.platform === "win32") {
    try { Bun.spawnSync(["taskkill", "/pid", String(pid), "/t", "/f"]); } catch {}
  } else {
    try { process.kill(-pid, "SIGKILL"); } catch { try { process.kill(pid, "SIGKILL"); } catch {} }
  }
}

export async function bashTool(
  args: { cmd: string; cwd?: string; timeout?: number; background?: boolean },
  ctx: ToolCtx,
): Promise<string> {
  if (typeof args.cmd !== "string" || !args.cmd.trim()) {
    throw new ToolError("bash requires {cmd}");
  }
  if (ctx.config.bashGuard !== "off") {
    const danger = scanDangerous(args.cmd);
    if (danger) {
      const logPath = join(ctx.config.dataDir, "blocked-commands.jsonl");
      try {
        mkdirSync(ctx.config.dataDir, { recursive: true });
        appendFileSync(logPath, JSON.stringify({
          at: new Date().toISOString(),
          sessionId: ctx.config.sessionId ?? null,
          cwd: ctx.cwd,
          cmd: args.cmd,
          reason: danger,
        }) + "\n");
      } catch {}
      ctx.warn?.(`dangerous command blocked (${danger}) — logged to ${logPath}; share that log with your model provider`);
      throw new ToolError(`dangerous command blocked: ${danger}. Do not retry it or attempt workarounds.`);
    }
    // Cloud-metadata / link-local URLs: fetch blocks these; bash must too.
    const meta = scanBlockedFetchUrls(args.cmd);
    if (meta.length) {
      throw new ToolError(
        `dangerous command blocked: cloud metadata / link-local URL (${meta.join(", ")}). Do not retry it or attempt workarounds.`,
      );
    }
    // Network egress policy (network:"deny" / allowlist). Best-effort in the
    // in-process guardrail; sandbox:"container" enforces it at the OS level.
    const egress = scanEgressPolicy(args.cmd, ctx.config);
    if (egress.length) {
      throw new ToolError(
        `denied by network policy: ${egress.join(", ")}. ${ctx.config.network === "deny"
          ? "Network egress is disabled for this workspace."
          : "This host is not in the network allowlist."} (sandbox:\"container\" enforces this at the OS level.)`,
      );
    }
  }
  if (ctx.config.sandbox !== "off") {
    const { outside, priv } = scanCmdPaths(args.cmd + (args.cwd ? ` ${args.cwd}` : ""), ctx);
    if (priv.length) {
      throw new ToolError(
        `denied: command references AP-private or restricted data-dir paths (${priv.join(", ")}) — sessions, checkpoints, credentials, and non-memory/artifacts dataDir paths are never accessible. Stay within ${ctx.cwd}.`,
      );
    }
    if (outside.length) {
      const ok = await ctx.permit({
        action: "bash outside workspace",
        detail: `${args.cmd.slice(0, 120)} → ${outside.join(", ")}`,
        path: outside[0],
      });
      if (!ok) {
        throw new ToolError(
          `denied: command references paths outside the workspace (${outside.join(", ")}) — do not explore unrelated folders; work within ${ctx.cwd}` +
          ` (the user can approve interactively, or pass --allow-outside in headless mode)`,
        );
      }
    }
  }
  // Soft git rails (push / remote add) are enforced in agent.ts so a
  // permission.bash "allow" rule can still permit them without a prompt.
  const cwd = args.cwd ? resolvePath(args.cwd, ctx.cwd) : ctx.cwd;
  const prefix = shellPrefix(ctx.config.shell);

  // Container sandbox: the command runs inside a throwaway docker/podman
  // container with only the workspace mounted and egress off by default — a
  // real OS boundary rather than a pattern scan. Background is not supported
  // in this mode (the container is torn down with --rm when the call returns).
  if (ctx.config.sandbox === "container") {
    if (args.background) {
      throw new ToolError("background:true is not supported under sandbox:\"container\" — run it in the foreground.");
    }
    const argv = containerArgv(ctx.config, args.cmd, cwd);
    const timeoutMs = Math.min((args.timeout ?? 120) * 1000 || DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS);
    // No agentChildEnv here: the container is the isolation, and the marker is
    // only meaningful to a host `ap` process (which cannot run inside the image).
    return spawnCaptured(argv, undefined, timeoutMs, ctx.signal);
  }

  if (args.background) {
    const logDir = join(ctx.config.dataDir, "logs");
    mkdirSync(logDir, { recursive: true });
    // Retention: background logs can contain sensitive command output — prune
    // anything older than 7 days whenever a new background process starts.
    try {
      const cutoff = Date.now() - 7 * 24 * 3600 * 1000;
      for (const f of readdirSync(logDir)) {
        if (f.startsWith("bg-") && (f.endsWith(".log") || f.endsWith(".sh") || f.endsWith(".ps1") || f.endsWith(".bat") || f.endsWith(".wrap.ps1"))
          && statSync(join(logDir, f)).mtimeMs < cutoff) {
          rmSync(join(logDir, f), { force: true });
        }
      }
    } catch {}
    const stamp = Date.now();
    const logFile = join(logDir, `bg-${stamp}.log`);
    // Never interpolate model cmd into a redirect wrapper — a `}` / `)` in
    // cmd can close the group and escape the log. Write cmd to a script we
    // own; the spawn argv only references paths under logDir.
    const isPs = prefix[0]!.toLowerCase().includes("powershell");
    const isCmd = /(?:^|[\\/])cmd(?:\.exe)?$/i.test(prefix[0]!);
    let child;
    if (isPs) {
      const body = join(logDir, `bg-${stamp}.ps1`);
      const wrap = join(logDir, `bg-${stamp}.wrap.ps1`);
      writeFileSync(body, args.cmd, "utf8");
      // Wrapper paths are ours; model text never appears in the -File argv.
      writeFileSync(wrap, `& '${body.replace(/'/g, "''")}' *> '${logFile.replace(/'/g, "''")}'\n`, "utf8");
      child = spawn(prefix[0]!, ["-NoProfile", "-File", wrap], {
        cwd, detached: true, stdio: "ignore", windowsHide: true, env: agentChildEnv(),
      });
    } else if (isCmd) {
      const body = join(logDir, `bg-${stamp}.bat`);
      writeFileSync(body, args.cmd, "utf8");
      // call "body" > "log" 2>&1 — only our paths in the /c string.
      child = spawn("cmd", ["/c", `call "${body}" > "${logFile}" 2>&1`], {
        cwd, detached: true, stdio: "ignore", windowsHide: true, env: agentChildEnv(),
      });
    } else {
      const body = join(logDir, `bg-${stamp}.sh`);
      writeFileSync(body, args.cmd, "utf8");
      // bash -c with $1/$2 so model text is never in the -c string.
      child = spawn(prefix[0]!, ["-c", `. "$1" >"$2" 2>&1`, "_", body, logFile], {
        cwd, detached: true, stdio: "ignore", windowsHide: true, env: agentChildEnv(),
      });
    }
    child.unref();
    // Register it so /ps (and `ap ps`) can find, tail, and kill it later —
    // detached children outlive this process and would otherwise be orphans.
    if (!ctx.config.light && child.pid) {
      const { recordBackground } = await import("../bg.ts");
      recordBackground(ctx.config, {
        pid: child.pid,
        cmd: args.cmd.replace(/\s+/g, " ").slice(0, 200),
        cwd,
        log: logFile,
        sessionId: ctx.config.sessionId ?? null,
      });
    }
    return `started background process pid=${child.pid} log=${logFile}`;
  }

  const timeoutMs = Math.min((args.timeout ?? 120) * 1000 || DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS);
  // agentChildEnv marks the child (and everything it spawns) as model-controlled,
  // so `ap trust accept` refuses there — the file-tool denies around the trust
  // store are meaningless if the model can just shell out to the CLI.
  return spawnCaptured([...prefix, args.cmd], cwd, timeoutMs, ctx.signal, agentChildEnv());
}

/** Spawn a command, cap+return its combined output, honouring timeout + abort. */
async function spawnCaptured(
  argv: string[],
  cwd: string | undefined,
  timeoutMs: number,
  signal: AbortSignal,
  env?: Record<string, string>,
): Promise<string> {
  const proc = Bun.spawn(argv, {
    ...(cwd ? { cwd } : {}),
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
    windowsHide: true,
    ...(env ? { env } : {}),
  } as any);

  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; killTree(proc.pid); }, timeoutMs);
  const onAbort = () => killTree(proc.pid);
  signal.addEventListener("abort", onAbort, { once: true });

  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      readSpawnCapped(proc.stdout, MAX_OUTPUT_BYTES * 4),
      readSpawnCapped(proc.stderr, MAX_OUTPUT_BYTES * 2),
      proc.exited,
    ]);
    let out = stdout;
    if (stderr.trim()) out += (out ? "\n" : "") + stderr;
    out = truncateMiddle(out.trim(), MAX_OUTPUT_BYTES);
    if (timedOut) return `[timed out after ${timeoutMs / 1000}s — process killed]\n${out}`;
    if (exitCode !== 0) return `[exit code ${exitCode}]\n${out}`;
    return out || "(no output)";
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}

/** Drain a spawn pipe up to `limit` bytes, then kill further reads. */
async function readSpawnCapped(stream: ReadableStream<Uint8Array> | number | null | undefined, limit: number): Promise<string> {
  if (!stream || typeof stream === "number") return "";
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      const room = limit - total;
      if (room <= 0) break;
      if (value.byteLength <= room) {
        chunks.push(value);
        total += value.byteLength;
      } else {
        chunks.push(value.subarray(0, room));
        total += room;
        break;
      }
    }
  } finally {
    try { await reader.cancel(); } catch {}
  }
  const buf = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.byteLength; }
  return new TextDecoder().decode(buf);
}
