#!/usr/bin/env bun
// Entry point: parse argv, dispatch to a mode via lazy import so the hot path
// (startup → first prompt) loads only what it needs.

const VERSION = "0.1.13";

export interface CliFlags {
  provider?: string;
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  cwd?: string;
  session?: string;
  mode?: string;
  resume?: string;
  continue?: boolean;
  json?: boolean;
  port?: number;
  prompt?: string;
  system?: string;
  noSandbox?: boolean;
  allowOutside?: boolean;
  light?: boolean;
  check?: string[];
  max?: number;
  agent?: string;
  effort?: string;
}

function parseArgs(argv: string[]): { cmd: string; flags: CliFlags; rest: string[] } {
  const flags: CliFlags = {};
  const rest: string[] = [];
  let cmd = "";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    switch (a) {
      case "--provider": flags.provider = argv[++i]; break;
      case "--model": case "-m": flags.model = argv[++i]; break;
      case "--base-url": flags.baseUrl = argv[++i]; break;
      case "--api-key": flags.apiKey = argv[++i]; break;
      case "--cwd": flags.cwd = argv[++i]; break;
      case "--session": flags.session = argv[++i]; break;
      case "--mode": flags.mode = argv[++i]; break;
      case "--plan": flags.mode = "plan"; break;
      case "--resume": flags.resume = argv[++i]; break;
      case "--continue": case "-c": flags.continue = true; break;
      case "--json": flags.json = true; break;
      case "--port": flags.port = Number(argv[++i]); break;
      case "--prompt": case "-p": flags.prompt = argv[++i]; break;
      case "--system": flags.system = argv[++i]; break;
      case "--no-sandbox": flags.noSandbox = true; break;
      case "--allow-outside": flags.allowOutside = true; break;
      case "--light": flags.light = true; break;
      case "--check": (flags.check ??= []).push(argv[++i]!); break;
      case "--max": flags.max = Number(argv[++i]); break;
      case "--agent": flags.agent = argv[++i]; break;
      case "--effort": flags.effort = argv[++i]; break;
      case "--version": case "-v": console.log(VERSION); process.exit(0);
      case "--help": case "-h": printHelp(); process.exit(0);
      default:
        if (!cmd && !a.startsWith("-")) cmd = a;
        else rest.push(a);
    }
  }
  return { cmd, flags, rest };
}

const HELP_TOPICS: Record<string, string> = {
  run: `ap run -p "task" [flags] — one-shot headless run

  Executes the task, prints the final answer to stdout (tool progress on
  stderr), exits. Ideal for scripts and CI.

  --json            one AgentEvent per line (NDJSON) on stdout instead
  --allow-outside   permit writes outside the workspace (otherwise denied)
  --agent <name>    run as a named agent profile (.ap/agents/<name>.md:
                    frontmatter description/model/tools, body = role)
  --session <id>    continue an existing session
  --system <text>   replace the system prompt for this run

  examples:
    ap run -p "explain the build pipeline" --plan
    ap run -p "bump all deps and fix breakage" --json | jq -r 'select(.type=="text").delta'`,

  loop: `ap loop -p "goal" [--check "cmd"]... [--max N] [--json] — run until DONE

  Loops work → verify until the goal is verifiably complete, not merely
  claimed complete. Each iteration: (1) work turn, (2) every --check command
  must exit 0 — failures are fed straight back with their output (no model
  tokens wasted), (3) an auditor pass re-verifies the original goal with
  fresh tool calls and either outputs LOOP_DONE or a gap list that becomes
  the next work order.

  Stops when: audit says LOOP_DONE and all checks pass · --max reached
  (exit 2) · the same gaps repeat with zero progress = stalled (exit 2) ·
  goal doesn't apply here → LOOP_BLOCKED (exit 3) · ctrl+c (exit 130).
  Context auto-compacts, so it can run indefinitely.

  After every iteration a collapsed per-file diff is printed (file + added/
  removed line counts), and a cumulative "total" when the loop ends.

  --check "cmd"     objective completion gate; repeatable, all must pass
  --max <n>         iteration cap (default: unlimited)

  examples:
    ap loop -p "make all tests pass" --check "bun test"
    ap loop -p "port utils/ to TypeScript" --check "bun x tsc --noEmit" --check "bun test" --max 20`,

  skills: `ap skills [add <src> [--skill name] | remove <name>] — SKILL.md packs

  Skills are reusable instruction folders (skills.sh / Claude Code format:
  SKILL.md with name/description frontmatter). Discovered from:
    <project>/.ap/skills/   <project>/.claude/skills/
    <dataDir>/skills/       ~/.claude/skills/   (first match wins)
  Available skills are listed one line each in the system prompt; the agent
  reads a skill's SKILL.md only when the task matches (cheap prompts).

  ap skills                          list installed skills
  ap skills add owner/repo           install every skill in a GitHub repo
  ap skills add owner/repo --skill x install one skill
  ap skills add https://www.skills.sh/owner/repo/name   same, from skills.sh URL
  ap skills remove <name>            delete from <dataDir>/skills

  Zero-dep installer: GitHub tree API + raw downloads — no git, no npx.`,

  mcp: `ap mcp — Model Context Protocol servers (the open plug-in standard)

  Any existing MCP server (GitHub, Postgres, Slack, browsers, …) plugs into
  AP: its tools appear to the model automatically as mcp_<server>_<tool>.
  Servers connect lazily before the first turn; a dead server degrades to a
  warning. Tools with readOnlyHint also work in plan mode. Not in --light.

  Config — same JSON as Claude Code, in either place (paste and go):
    .mcp.json in the project root:            {"mcpServers": {"name": {…}}}
    mcpServers block in ap.config.json or <dataDir>/config.json
  stdio: {"command": "bun", "args": ["x", "@modelcontextprotocol/server-filesystem", "."]}
  http:  {"url": "https://example.com/mcp", "headers": {"authorization": "Bearer …"}}

  ap mcp                                 list servers, status, tools (exit 1 if any down)
  ap mcp call <server> <tool> '<json>'   call one tool directly, no LLM (testing)
  ap mcp add <name> <command> [args...]  add a stdio server to <dataDir>/config.json
  ap mcp add <name> --url <url>          add a Streamable HTTP server
  ap mcp add ... --project               write to .mcp.json in this project instead
  ap mcp remove <name>                   remove from either config file

  If a server command needs flags that collide with ap's own (-m, --port, …),
  add the entry to the JSON file directly instead of via "ap mcp add".`,

  serve: `ap serve [--port 4141] — HTTP server mode

  POST /session {cwd?}                    → {id}
  POST /session/:id/message {text, ...}   → blocks → {text, messages}
  GET  /session/:id/events                → SSE stream of AgentEvents
  GET  /health · GET /session/:id/messages · DELETE /session/:id`,

  acp: `ap acp — Agent Client Protocol server (Zed and other ACP editors)

  Speaks ACP v1 over stdio: the editor spawns ap and drives it; AP streams
  message / thought / tool-call updates back live. Sandbox permission
  requests appear as native permission dialogs in the editor, plan/code are
  exposed as ACP session modes, sessions persist (loadSession supported),
  and MCP servers configured in the editor are passed through to AP's own
  MCP client. Editor "cancel" aborts the turn cleanly.

  Zed — add to settings.json, then open the Agent Panel and pick "AP":
    { "agent_servers": { "AP": { "command": "ap", "args": ["acp"] } } }

  Flags carry through: ap acp --provider x -m model --light etc.
  Hooks fire here too (hooks.onDone/onError — command or webhook URL).`,

  sessions: `ap sessions — list · ap sessions search <q> — ripgrep transcripts
  ap resume — interactive picker · ap -c — resume most recent
  Sessions are append-only JSONL in <dataDir>/sessions/<id>.jsonl.`,
};

function printHelp(topic?: string) {
  if (topic) {
    const t = HELP_TOPICS[topic];
    console.log(t ?? `no extra help for "${topic}" — topics: ${Object.keys(HELP_TOPICS).join(", ")}`);
    return;
  }
  console.log(`AP (Agent Platform) ${VERSION} — minimal fast coding agent
zero deps · ~45ms start · OpenAI-compatible providers · https://github.com/Sanjay-doppalapudi/Agent-Platform

Usage:
  ap                       interactive REPL in the current directory
  ap run -p "task"         one-shot run (--json for NDJSON events)
  ap loop -p "goal"        loop work→verify until the goal is verifiably done
  ap serve [--port 4141]   HTTP server mode (sessions + SSE)
  ap acp                   ACP agent for editors (Zed): stdio, modes, permissions
  ap skills [add|remove]   list/install SKILL.md packs (skills.sh compatible)
  ap mcp [add|call|...]    connect MCP servers — their tools become agent tools
  ap models [query]        search the models.dev catalog (context + prices)
  ap auth <provider>       store an API key (hidden input, user-locked file)
  ap resume                pick a recent session to resume
  ap sessions [search q]   list sessions / full-text search them
  ap share [id]            export a transcript as one self-contained HTML file
  ap ps [tail|kill <pid>]  background processes started with bash background:true
  ap doctor [--offline]    diagnose the environment (deps, keys, endpoint, MCP)
  ap prompt [--cwd dir]    print the exact system prompt for a directory
  ap tool <name> '<json>'  run one tool directly, no LLM (testing)
  ap help <command>        detailed help: ${Object.keys(HELP_TOPICS).join(" · ")}

Provider / model:
  --provider <name>    named provider from config
  -m, --model <id>     model override for this invocation
  --base-url <url>     any OpenAI-compatible endpoint (pair with --api-key)
  --api-key <key>      key override (prefer: ap auth <provider>)

Behavior:
  --mode <plan|code>   plan = read-only tools + produce a plan (also: --plan)
  --effort <level>     reasoning effort low|medium|high (config: reasoningEffort)
  --light              minimal profile: core 6 tools, smallest prompt, no extras
  --cwd <dir>          working directory for the agent
  --session <id>       attach to a session   --resume <id> / -c   resume (REPL)

Loop mode:
  --check "cmd"        objective completion gate (repeatable, all must exit 0)
  --max <n>            iteration cap (default unlimited)

Safety:
  --no-sandbox         disable the workspace write-sandbox this invocation
  --allow-outside      headless runs: allow writes outside the workspace

Output:
  --json               NDJSON AgentEvents on stdout (run/loop)
  -v, --version        print version        -h, --help           this screen

Examples:
  ap                                          # chat in the current repo
  ap run -p "add input validation to signup"
  ap loop -p "make bun test pass" --check "bun test"
  ap skills add vercel-labs/agent-skills --skill web-design-guidelines
  ap -m deepseek-v4-pro --plan                # plan mode on another model

Keys in the REPL: / commands · @ file picker · ctrl+o details · ctrl+c abort`);
}

const { cmd, flags, rest } = parseArgs(process.argv.slice(2));

try {
  await dispatch(cmd, flags, rest);
} catch (e) {
  console.error(`error: ${(e as Error).message}`);
  // Double-click launch: the console window vanishes on exit — hold it open
  // so the error is readable. Piped/scripted runs (not a TTY) are unaffected.
  if (process.stdin.isTTY && process.stdout.isTTY) {
    console.error("\npress Enter to exit...");
    const { createInterface } = await import("node:readline");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    await new Promise<void>((res) => rl.question("", () => { rl.close(); res(); }));
  }
  process.exit(1);
}

async function dispatch(cmd: string, flags: CliFlags, rest: string[]) {
switch (cmd) {
  case "run": {
    const { runMain } = await import("./run.ts");
    await runMain(flags);
    break;
  }
  case "loop": {
    const { loopMain } = await import("./loop.ts");
    await loopMain(flags);
    break;
  }
  case "skills": {
    const { loadConfig } = await import("./config.ts");
    const config = loadConfig(flags);
    const { discoverSkills, parseSource, listRemoteSkills, installSkill, removeSkill } = await import("./skills.ts");
    const sub = rest[0];
    if (!sub) {
      const skills = discoverSkills(config);
      if (!skills.length) { console.log("no skills installed — try: ap skills add <owner>/<repo>"); break; }
      for (const s of skills) console.log(`${s.name}  [${s.source}]  ${s.description.slice(0, 100)}\n  ${s.path}`);
      break;
    }
    if (sub === "add" && rest[1]) {
      const src = parseSource(rest[1]);
      // --skill after "add" arrives in rest (not a global flag)
      const skillFlagIdx = rest.indexOf("--skill");
      const only = skillFlagIdx >= 0 ? rest[skillFlagIdx + 1] : src.skill;
      const remote = await listRemoteSkills(src.owner, src.repo);
      if (!remote.length) { console.error(`no SKILL.md found in ${src.owner}/${src.repo}`); process.exit(1); }
      const targets = only ? remote.filter((r) => r.name === only) : remote;
      if (!targets.length) {
        console.error(`skill "${only}" not in ${src.owner}/${src.repo} — available: ${remote.map((r) => r.name).join(", ")}`);
        process.exit(1);
      }
      for (const t of targets) {
        console.log(`installing ${t.name} from ${src.owner}/${src.repo}…`);
        const dest = await installSkill(config.dataDir, src.owner, src.repo, t, (l) => console.log(l));
        console.log(`✓ ${t.name} → ${dest}`);
      }
      break;
    }
    if (sub === "remove" && rest[1]) {
      console.log(removeSkill(config.dataDir, rest[1]) ? `removed ${rest[1]}` : `not found in ${config.dataDir}\\skills: ${rest[1]}`);
      break;
    }
    console.error(`usage: ap skills | ap skills add <owner>/<repo> [--skill name] | ap skills remove <name>`);
    process.exit(1);
    break;
  }
  case "mcp": {
    const { loadConfig } = await import("./config.ts");
    const config = loadConfig(flags);
    const { initMcp, mcpStatus, mcpServerSpecs } = await import("./mcp.ts");
    const warn = (m: string) => console.error(`⚠ ${m}`);
    const sub = rest[0];

    if (!sub || sub === "list") {
      const specs = mcpServerSpecs(config);
      if (!Object.keys(specs).length) {
        console.log(`no MCP servers configured.
Add one:  ap mcp add <name> <command> [args...]   (stdio server)
          ap mcp add <name> --url <https://...>   (Streamable HTTP server)
Or drop a Claude-Code-format .mcp.json in the project root — works as-is.
Example:  ap mcp add fs bun x @modelcontextprotocol/server-filesystem .`);
        break;
      }
      await initMcp(config, warn);
      let anyFailed = false;
      for (const s of mcpStatus()) {
        if (!s.ok) anyFailed = true;
        const head = `${s.ok ? "✓" : "✗"} ${s.name}  [${s.transport}]${s.serverName ? `  ${s.serverName}` : ""}`;
        console.log(s.ok ? `${head}  ·  ${s.tools.length} tools` : `${head}  ·  ${s.error}`);
        for (const t of s.tools) {
          console.log(`    ${t.canonical}${t.readOnly ? "  (read-only)" : ""}  ${t.description.replace(/\s+/g, " ").slice(0, 80)}`);
        }
      }
      process.exit(anyFailed ? 1 : 0);
    }

    if (sub === "call" && rest[1] && rest[2]) {
      await initMcp(config, warn);
      const { execTool } = await import("./tools/index.ts");
      const start = performance.now();
      const { output, error } = await execTool(`${rest[1]}.${rest[2]}`, rest[3] ?? "{}", {
        cwd: config.cwd,
        signal: new AbortController().signal,
        config,
        permit: async () => true, // developer typed the exact args — trusted
        warn,
      });
      console.log(output);
      console.error(`[${rest[1]}/${rest[2]} ${error ? "error" : "ok"} in ${Math.round(performance.now() - start)}ms]`);
      process.exit(error ? 1 : 0);
    }

    if (sub === "add" && rest[1]) {
      const name = rest[1]!;
      let url: string | undefined;
      let project = false;
      const cmdParts: string[] = [];
      for (let i = 2; i < rest.length; i++) {
        if (rest[i] === "--url") url = rest[++i];
        else if (rest[i] === "--project") project = true;
        else cmdParts.push(rest[i]!);
      }
      const spec = url
        ? { url }
        : cmdParts.length
          ? { command: cmdParts[0]!, ...(cmdParts.length > 1 ? { args: cmdParts.slice(1) } : {}) }
          : null;
      if (!spec) {
        console.error(`usage: ap mcp add <name> <command> [args...] | ap mcp add <name> --url <url>  [--project]`);
        process.exit(1);
      }
      const { existsSync, readFileSync, writeFileSync } = await import("node:fs");
      const { join } = await import("node:path");
      const file = project ? join(config.cwd, ".mcp.json") : join(config.dataDir, "config.json");
      let cur: any = {};
      if (existsSync(file)) {
        try { cur = JSON.parse(readFileSync(file, "utf8")); } catch {
          console.error(`refusing to overwrite unparseable ${file} — fix it first`);
          process.exit(1);
        }
      }
      cur.mcpServers = { ...cur.mcpServers, [name]: spec };
      writeFileSync(file, JSON.stringify(cur, null, 2) + "\n");
      console.log(`added mcp server "${name}" → ${file}\nverify with: ap mcp list`);
      break;
    }

    if (sub === "remove" && rest[1]) {
      const { existsSync, readFileSync, writeFileSync } = await import("node:fs");
      const { join } = await import("node:path");
      let removed = false;
      for (const file of [join(config.cwd, ".mcp.json"), join(config.dataDir, "config.json")]) {
        if (!existsSync(file)) continue;
        try {
          const cur = JSON.parse(readFileSync(file, "utf8"));
          if (cur.mcpServers?.[rest[1]!]) {
            delete cur.mcpServers[rest[1]!];
            writeFileSync(file, JSON.stringify(cur, null, 2) + "\n");
            console.log(`removed "${rest[1]}" from ${file}`);
            removed = true;
          }
        } catch {}
      }
      if (!removed) { console.error(`server "${rest[1]}" not found in .mcp.json or ${config.dataDir}\\config.json`); process.exit(1); }
      break;
    }

    console.error(`usage: ap mcp [list] | ap mcp call <server> <tool> '<json>' | ap mcp add <name> <command...> [--url u] [--project] | ap mcp remove <name>`);
    process.exit(1);
    break;
  }
  case "doctor": {
    const { loadConfig } = await import("./config.ts");
    const { runChecks, overallState } = await import("./doctor.ts");
    const { currentTheme, paint, setTheme } = await import("./theme.ts");
    const config = loadConfig(flags);
    if (config.theme) setTheme(config.theme);
    const t = currentTheme();
    const noNet = rest.includes("--offline");
    console.log(`AP ${VERSION} · ${process.platform} · Bun ${Bun.version}\n`);
    const checks = await runChecks(config, { net: !noNet, mcp: !noNet });
    for (const c of checks) {
      const mark = c.state === "ok" ? paint(t.success, "✓") : c.state === "warn" ? paint(t.warn, "!") : paint(t.error, "✗");
      console.log(`${mark} ${c.name.padEnd(12)} ${c.detail}`);
      if (c.fix) console.log(`  ${paint(t.dim, `→ ${c.fix}`)}`);
    }
    const state = overallState(checks);
    const bad = checks.filter((c) => c.state !== "ok").length;
    console.log(
      state === "ok"
        ? `\n${paint(t.success, "all checks passed")} — you're ready: run "ap" in a project directory`
        : `\n${bad} check${bad === 1 ? " needs" : "s need"} attention (see → lines above)`,
    );
    process.exit(state === "fail" ? 1 : 0);
  }
  case "ps": {
    const { loadConfig } = await import("./config.ts");
    const { listBackground, tailLog, killBackground, formatBytes } = await import("./bg.ts");
    const config = loadConfig(flags);
    const sub = rest[0];
    if (sub === "kill" && rest[1]) {
      const r = killBackground(config, Number(rest[1]));
      console.log(r.message);
      process.exit(r.ok ? 0 : 1);
    }
    const rows = listBackground(config);
    if (sub === "tail") {
      const pick = rest[1] ? rows.find((r) => r.pid === Number(rest[1])) : rows.find((r) => r.alive) ?? rows[0];
      if (!pick) { console.error(rest[1] ? `no background process with pid ${rest[1]}` : "no background processes"); process.exit(1); }
      console.error(`— ${pick.cmd} (pid ${pick.pid}) · ${pick.log} —`);
      console.log(tailLog(pick.log, Number(rest[2]) || 40));
      break;
    }
    if (!rows.length) { console.log("no background processes"); break; }
    for (const r of rows) {
      console.log(`${r.alive ? "●" : "○"} ${r.pid}  ${r.cmd.slice(0, 60)}  ${r.alive ? "running" : "exited"} · ${formatBytes(r.bytes)} · ${r.log}`);
    }
    console.log(`\nap ps tail <pid> [lines] · ap ps kill <pid>`);
    break;
  }
  case "share": {
    const { loadConfig } = await import("./config.ts");
    const { Session } = await import("./session.ts");
    const { exportSessionHtml } = await import("./shareview.ts");
    const { openInBrowser } = await import("./planview.ts");
    const config = loadConfig(flags);
    const id = rest[0] ?? Session.latest(config.dataDir);
    if (!id) { console.error("no sessions yet"); process.exit(1); }
    const session = Session.load(config.dataDir, id);
    if (!session.history.length) { console.error(`session ${id} is empty`); process.exit(1); }
    const p = exportSessionHtml(config.dataDir, id, session.history, "", config.cwd);
    console.log(p);
    openInBrowser(p);
    break;
  }
  case "help": {
    printHelp(rest[0]);
    break;
  }
  case "serve": {
    const { serveMain } = await import("./server.ts");
    await serveMain(flags);
    break;
  }
  case "acp": {
    const { acpMain } = await import("./acp.ts");
    await acpMain(flags);
    break;
  }
  case "tool": {
    const { toolMain } = await import("./tool-cli.ts");
    await toolMain(flags, rest);
    break;
  }
  case "prompt": {
    const { loadConfig } = await import("./config.ts");
    const { buildSystemPrompt } = await import("./prompt.ts");
    const sys = buildSystemPrompt(loadConfig(flags));
    console.log(sys);
    console.error(`\n[${sys.length} chars, ~${Math.round(sys.length / 4)} tokens]`);
    break;
  }
  case "models": {
    const { loadConfig } = await import("./config.ts");
    const { loadCatalog, searchModels } = await import("./models.ts");
    const config = loadConfig(flags);
    const catalog = await loadCatalog(config.dataDir);
    const rows = searchModels(catalog, rest.join(" "));
    if (!rows.length) { console.log("no matches"); break; }
    for (const r of rows) {
      const cost = r.inCost != null ? `$${r.inCost}/$${r.outCost ?? "?"} per M` : "";
      const ctx = r.ctx ? `${Math.round(r.ctx / 1000)}k ctx` : "";
      console.log(`${r.provider}/${r.model}  ${[ctx, cost].filter(Boolean).join(" · ")}`);
    }
    break;
  }
  case "auth": {
    const { loadConfig } = await import("./config.ts");
    const { listKeys, setKey } = await import("./creds.ts");
    const config = loadConfig(flags);
    const [prov, keyArg] = rest;
    if (!prov) {
      const stored = listKeys(config.dataDir);
      console.log(stored.length ? `stored keys: ${stored.join(", ")}` : "no stored keys — usage: harness auth <provider> [key]");
      break;
    }
    let key = keyArg;
    if (!key) {
      const { emitKeypressEvents } = await import("node:readline");
      emitKeypressEvents(process.stdin);
      const { readSecret } = await import("./input.ts");
      key = await readSecret(`API key for ${prov}: `);
    }
    if (!key) { console.error("no key entered"); process.exit(1); }
    setKey(config.dataDir, prov, key);
    console.log(`stored key for "${prov}" in ${config.dataDir}\\credentials.json (user-only access)`);
    break;
  }
  case "resume": {
    const { loadConfig } = await import("./config.ts");
    const { Session } = await import("./session.ts");
    const config = loadConfig(flags);
    const sessions = Session.list(config.dataDir, 15);
    if (!sessions.length) { console.log("no sessions yet"); break; }
    sessions.forEach((s, i) => console.log(`${i + 1}. ${s.id}  ${new Date(s.mtime).toLocaleString()}`));
    const { createInterface } = await import("node:readline");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    const answer = await new Promise<string>((res) => rl.question("resume which? [1] ", res));
    rl.close();
    const pick = sessions[Math.max(1, Number(answer) || 1) - 1];
    if (!pick) { console.error("no such entry"); process.exit(1); }
    flags.resume = pick.id;
    const { replMain } = await import("./repl.ts");
    await replMain(flags);
    break;
  }
  case "sessions": {
    const { loadConfig } = await import("./config.ts");
    const { Session } = await import("./session.ts");
    const { join } = await import("node:path");
    const config = loadConfig(flags);
    if (rest[0] === "search" && rest[1]) {
      const query = rest.slice(1).join(" ");
      const dir = join(config.dataDir, "sessions");
      const p = Bun.spawnSync(["rg", "-i", "-l", "--no-messages", query, dir], { stdout: "pipe", stderr: "ignore" });
      const files = (p.stdout?.toString() ?? "").split("\n").filter(Boolean);
      if (!files.length) { console.log("no sessions match"); break; }
      for (const f of files.slice(0, 20)) {
        const id = f.replace(/\\/g, "/").split("/").pop()!.replace(/\.jsonl$/, "");
        const m = Bun.spawnSync(["rg", "-i", "-m", "1", "-o", `.{0,40}${query}.{0,40}`, f], { stdout: "pipe", stderr: "ignore" });
        console.log(`${id}  ${(m.stdout?.toString() ?? "").split("\n")[0]?.trim() ?? ""}`);
      }
      console.log(`\nresume with: ap --resume <id>`);
      break;
    }
    for (const s of Session.list(config.dataDir, 20)) {
      console.log(`${s.id}  ${new Date(s.mtime).toLocaleString()}`);
    }
    break;
  }
  case "": {
    const { replMain } = await import("./repl.ts");
    await replMain(flags);
    break;
  }
  default:
    console.error(`unknown command: ${cmd} (try --help)`);
    process.exit(1);
}
}
