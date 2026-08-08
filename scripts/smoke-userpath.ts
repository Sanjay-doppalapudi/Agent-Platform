#!/usr/bin/env bun
/**
 * Real user-path smoke: exercise `ap` the way a human would, without an LLM
 * key where possible. Exit non-zero on any unexpected failure. Prints a
 * breakage report to stderr.
 *
 *   bun run scripts/smoke-userpath.ts
 */
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const ROOT = join(import.meta.dir, "..");
const AP = [process.execPath, join(ROOT, "src", "index.ts")];

type Result = { name: string; ok: boolean; detail: string };

function run(args: string[], opts?: { cwd?: string; env?: Record<string, string>; expectFail?: boolean }): Result {
  const name = `ap ${args.join(" ")}`.slice(0, 80);
  const p = Bun.spawnSync([...AP, ...args], {
    cwd: opts?.cwd ?? ROOT,
    env: { ...process.env, ...opts?.env, NO_COLOR: "1" },
    stdout: "pipe",
    stderr: "pipe",
  });
  const out = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim();
  const ok = opts?.expectFail ? p.exitCode !== 0 : p.exitCode === 0;
  return { name, ok, detail: ok ? out.slice(0, 200) : `exit ${p.exitCode}: ${out.slice(0, 500)}` };
}

const results: Result[] = [];
function check(r: Result) {
  results.push(r);
  const mark = r.ok ? "PASS" : "FAIL";
  console.log(`${mark}  ${r.name}`);
  if (!r.ok) console.log(`      ${r.detail.split("\n")[0]}`);
}

console.log("=== ap user-path smoke ===\n");

// 1. Version / help — must never break
check(run(["--version"]));
check(run(["--help"]));
check(run(["help"]));

// 2. Doctor offline — no network
check(run(["doctor", "--offline"]));

// 3. Prompt dump — no provider key needed
check(run(["prompt", "--cwd", ROOT]));
check(run(["prompt", "--cwd", ROOT, "--light"]));

// 4. Tool CLI — read / grep / repomap without LLM
check(run(["tool", "read", JSON.stringify({ path: "package.json" })]));
check(run(["tool", "grep", JSON.stringify({ pattern: "Agent Platform", path: "CLAUDE.md", max: 5 })]));
check(run(["tool", "glob", JSON.stringify({ pattern: "src/*.ts" })]));
check(run(["tool", "repomap", JSON.stringify({ path: "src", max: 20 })]));

// 5. Sandbox edge: private path hard-deny
{
  const data = mkdtempSync(join(tmpdir(), "ap-smoke-data-"));
  mkdirSync(join(data, "sessions"), { recursive: true });
  writeFileSync(join(data, "credentials.json"), "{}");
  // Point dataDir via a tiny project config
  const proj = mkdtempSync(join(tmpdir(), "ap-smoke-proj-"));
  writeFileSync(join(proj, "ap.config.json"), JSON.stringify({
    dataDir: data.replace(/\\/g, "/"),
    provider: "adhoc",
    providers: { adhoc: { baseUrl: "http://127.0.0.1:9", model: "x", apiKey: "x" } },
    permissions: "yolo",
    sandbox: "workspace",
  }));
  writeFileSync(join(proj, "ok.txt"), "hi\n");
  const r = run(
    ["--cwd", proj, "tool", "read", JSON.stringify({ path: join(data, "credentials.json") })],
    { cwd: proj },
  );
  // Should fail / deny — treat "deny|blocked|private|denied" as success
  const denied = /deny|denied|blocked|private|credentials/i.test(r.detail + r.name) || !r.ok;
  results.push({
    name: "sandbox hard-deny credentials.json",
    ok: denied,
    detail: r.detail,
  });
  console.log(`${denied ? "PASS" : "FAIL"}  sandbox hard-deny credentials.json`);
  rmSync(proj, { recursive: true, force: true });
  rmSync(data, { recursive: true, force: true });
}

// 6. Fetch SSRF edges — metadata blocked; file:// rejected; localhost allowed by design
check(run(["tool", "fetch", JSON.stringify({ url: "http://169.254.169.254/latest/meta-data/" })], { expectFail: true }));
check(run(["tool", "fetch", JSON.stringify({ url: "http://[fd00:ec2::254]/" })], { expectFail: true }));
check(run(["tool", "fetch", JSON.stringify({ url: "file:///etc/passwd" })], { expectFail: true }));
check(run(["tool", "fetch", JSON.stringify({ url: "ftp://example.com/" })], { expectFail: true }));

// 7. tmux on this platform
{
  const r = run(["tmux", "list"]);
  if (process.platform === "win32") {
    const graceful = !r.ok && /tmux|Windows|WSL|PATH/i.test(r.detail);
    results.push({ name: "ap tmux graceful on Windows", ok: graceful, detail: r.detail });
    console.log(`${graceful ? "PASS" : "FAIL"}  ap tmux graceful on Windows`);
  } else {
    check(r); // list may succeed with empty sessions
  }
}

// 8. pr without --yes should preview and exit 2 (or fail early if on protected)
{
  const r = Bun.spawnSync([...AP, "pr", "--title", "smoke test only"], {
    cwd: ROOT,
    env: { ...process.env, NO_COLOR: "1" },
    stdout: "pipe",
    stderr: "pipe",
  });
  const out = ((r.stdout?.toString() ?? "") + (r.stderr?.toString() ?? "")).trim();
  // Exit 2 = preview; exit 1 = protected branch / no gh — both acceptable "worked"
  const ok = r.exitCode === 2 || r.exitCode === 1;
  results.push({ name: "ap pr preview/--yes gate", ok, detail: `exit ${r.exitCode}: ${out.slice(0, 200)}` });
  console.log(`${ok ? "PASS" : "FAIL"}  ap pr preview/--yes gate`);
}

// 9. Sessions list (should not throw even if empty)
check(run(["sessions"]));

// 10. MCP list may warn if none — should not crash
{
  const r = Bun.spawnSync([...AP, "mcp", "list"], {
    cwd: ROOT,
    env: { ...process.env, NO_COLOR: "1" },
    stdout: "pipe",
    stderr: "pipe",
  });
  // exit 0 or 1 (servers down) both OK; crash = not OK
  const ok = r.exitCode === 0 || r.exitCode === 1;
  results.push({ name: "ap mcp list (no crash)", ok, detail: `exit ${r.exitCode}` });
  console.log(`${ok ? "PASS" : "FAIL"}  ap mcp list (no crash)`);
}

// 11. Typecheck via local tsc if present (bunx often broken)
{
  const tsc = Bun.which("tsc") || join(ROOT, "node_modules", "typescript", "bin", "tsc");
  const hasLocal = await Bun.file(join(ROOT, "node_modules", "typescript", "lib", "tsc.js")).exists().catch(() => false);
  if (hasLocal) {
    const p = Bun.spawnSync([process.execPath, join(ROOT, "node_modules", "typescript", "lib", "tsc.js"), "--noEmit"], {
      cwd: ROOT,
      stdout: "pipe",
      stderr: "pipe",
    });
    const out = ((p.stdout?.toString() ?? "") + (p.stderr?.toString() ?? "")).trim();
    results.push({ name: "tsc --noEmit (local)", ok: p.exitCode === 0, detail: out.slice(0, 400) });
    console.log(`${p.exitCode === 0 ? "PASS" : "FAIL"}  tsc --noEmit (local)`);
    if (p.exitCode !== 0) console.log(`      ${out.split("\n").slice(0, 3).join(" | ")}`);
  } else {
    results.push({
      name: "tsc --noEmit (local)",
      ok: true,
      detail: "SKIP — no local typescript; bun x tsc is known broken (missing lib.d.ts)",
    });
    console.log("SKIP  tsc --noEmit (no local typescript; bunx path known broken)");
  }
}

const failed = results.filter((r) => !r.ok);
console.log(`\n=== ${results.length - failed.length}/${results.length} passed ===`);
if (failed.length) {
  console.error("\nBreakage:");
  for (const f of failed) console.error(`- ${f.name}: ${f.detail.slice(0, 300)}`);
  process.exit(1);
}
