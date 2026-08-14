# Security Policy

## What AP is (threat model)

AP is a **local development tool** that executes commands, edits files, and fetches
URLs chosen by an LLM — inside a session that the local user starts, watches, and
can abort at any moment. It is *dual-use by design*: the same capability that fixes
your build can delete a file. AP is **not a privilege boundary**: it runs with the
invoking user's permissions, and its guardrails are engineering controls against
*accidents and model misbehavior*, not a substitute for OS-level isolation.

**For genuinely untrusted code or goals, use `sandbox: "container"`.** The default
`sandbox: "workspace"` is a guardrail, not a VM — its file-path containment holds
well, but `bash` is host code execution, so pattern scanning is best-effort and a
motivated model can escape it (e.g. a path decoded at runtime, or a socket opened
from a script). `sandbox: "container"` runs every `bash` command inside a throwaway
docker/podman container with **only the workspace bind-mounted** and **egress off by
default** — a real OS boundary. Independently, `network` (`"deny"` / hostname
allowlist) gates `fetch`/`websearch`/`bash` egress; cloud-metadata hosts are always
blocked regardless of policy.

## Guardrails (defense in depth)

| Layer | Mechanism |
|---|---|
| Write sandbox | `write`/`edit`/`bash` mutations outside the workspace (plus `dataDir/memory` and `dataDir/artifacts`) require an interactive user permit (`y/N/always`); headless runs auto-deny unless `--allow-outside`. The rest of `<dataDir>` is **not** freely writable |
| Sandbox modes (`sandbox`) | `"workspace"` (default): in-process path containment + bash pattern scan — a guardrail. `"container"` (`--sandbox container`): each `bash` runs in a throwaway docker/podman container, **only the workspace mounted** (`-v cwd:/workspace`), `--network none` unless `network:"allow"`, `--security-opt no-new-privileges`; credentials/sessions/other repos are structurally unreachable. `"off"`: no file gates. Image via `sandboxImage` (default `alpine`) |
| Network egress (`network`) | `undefined`/`"allow"` keeps egress open; `"deny"` blocks all; a `string[]` is a suffix-match hostname allowlist. Applied to `fetch`, `websearch`, and `bash` URL tokens (best-effort in `"workspace"` mode; OS-enforced by `--network none` in `"container"` mode). Cloud-metadata / link-local hosts are blocked under **every** policy |
| Read scoping | Reads outside the workspace + skills/memory/commands dirs also require a permit; bash commands are path-token scanned (incl. `../` escapes). Symlink targets are resolved before the containment check |
| AP-private data | Session transcripts, checkpoints, `credentials.json`, `config.json`, and the home-dir **workspace-trust store** (`~/.ap/trusted-workspaces.json`) are **hard-denied for read and write — permits cannot override** (via file tools and via bash path-token scan) |
| Windows filename equivalence | Path resolution mirrors the Win32 filesystem's trimming of **trailing dots and spaces** on every path component, so `ap.config.json.` / `credentials.json ` cannot address a guarded file while dodging the name-based hard-denies (win32 only; those bytes are preserved on POSIX) |
| Dangerous commands | Destructive patterns (recursive absolute deletes, disk format, registry edits, `curl \| sh`, `bash <(…)`, PowerShell IEX/encoded, `find -delete`, fork bombs, …) are blocked outright — never prompted — and logged to `<dataDir>/blocked-commands.jsonl`. Per-tool `permission.bash` patterns apply to **each segment** of compound commands (`&&`/`||`/`|`/`;`); the strictest verdict wins |
| Plan mode | Structurally read-only: mutating tool schemas are not even sent to the model, plus a runtime backstop for hallucinated calls. `/confirm edits on` additionally forces ask on every edit/write even under yolo |
| Hooks | Hook command strings come **only from the user's own config files**, never from model output; tool arguments are passed via environment variables, never interpolated into the command line (`afterEdit` also receives `AP_ARGS.paths`) |
| Credentials | Stored in `<dataDir>/credentials.json`, file-ACL'd to the invoking user (ACL apply failure fails `ap auth`); `.env` / common secret filenames are redacted in `read` **and** `grep` (lexical path **and** symlink target) |
| `ap serve` | Binds `127.0.0.1` by default. **Always** requires a bearer token (`--token` / `AP_SERVE_TOKEN`, auto-generated if omitted). `allowOutside` / `system` are rejected on HTTP. `/health` does not advertise provider/model |
| Fetch | Cloud metadata / link-local hosts (`169.254.0.0/16`, `metadata.google.internal`, `fd00:ec2::254`, …) are refused — including **IPv4 embedded in IPv6**. DNS is resolved and pinned; every redirect is validated. The same host policy is applied to **URL tokens in bash** |
| Bash path scan | `file://` and Windows UNC paths are treated as local paths (private-data gate). Compound `permission.bash` rules unwrap `bash/sh/cmd/powershell -c` wrappers and split on `&` |
| Workspace trust | Untrusted projects may only set a **safe allowlist** (`ignore`, `theme`, `showReasoning`, `parallelPolicy`). hooks/MCP/sandbox/bashGuard/redactEnv/permission(s)/providers/router/dataDir/confirmEdits/planModel/codeModel/resource limits/`shell`/`git` require `ap trust accept`. Untrusted `.mcp.json`, project workflows, and project custom commands are ignored. Project `dataDir` is never honored (use `AP_DATA_DIR` / home config). Untrusted provider entries cannot set `baseUrl`/`apiKey`/`apiKeyEnv`/headers |
| Granting trust | `ap trust accept` refuses in any process AP spawned for the model (`AP_AGENT_CHILD`) **and** outside an interactive TTY, then requires the user to type `trust` after seeing the resolved git root. The agent cannot self-escalate by shelling out to the CLI, and the system prompt does not name the command |
| Privileged project files | `ap.config.json` / `harness.config.json` / `.mcp.json` (in **any** directory the config walkers climb to), `.git/hooks/**`, and `.ap|.claude/commands/**` are **hard-denied for model writes — permits cannot override**. They grant code execution on the next run, and a trusted project config passes through wholesale |
| Prompt injection | Project notes/skills/agents/commands/workflows/DECISIONS load only when trusted. Model writes to `HARNESS.md` / `AP.md` / `.ap/skills|agents|workflows` / `DECISIONS.md` are hard-denied. Memory cards must be exactly three schema lines and are marked UNTRUSTED in the prompt; free-form memory is rejected on write and skipped on read |
| Artifacts | Model HTML gets a no-network CSP (`script-src 'none'`); meta refresh, `<script>`, `<iframe>`, and inline `on*` handlers are stripped; slug path segments are alphabet-validated and containment-asserted; files are created with exclusive (`wx`) opens |
| Real git | `/commit` and `/pr` never push; `git push --force`, `reset --hard`, `clean -f`, `remote set-url` are blocked even with `-C`/`-c` global options; plain `git push` / `remote add` require a permit unless **every** sensitive sub-command has an explicit non-`*` `permission.bash` allow — including pushes smuggled inside `$(…)` / backtick substitution, which are enumerated as first-class segments (not hidden behind an allowed outer part); the command is read through the same `cmd`/`command`/`script` aliases the tool executes, so no argument spelling skips the rail, and a bare `git.exe` verb is normalized so it cannot dodge the force-push block; `permission.bash` unwraps wrappers, strips `command`/`env`/`eval`/`sudo` prefixes, and reduces absolute binary paths to basename |
| Bash fetch | URL tokens in bash are host-policy scanned; `curl -L` / `wget -L` refused; plain `wget https://…` refused unless `--max-redirect=0` (wget follows redirects by default) |
| Sessions | IDs are CSPRNG UUIDs; path-unsafe ids are rejected on load |
| Resource limits | Per-command timeouts with process-tree kill; **bounded** reads of tool output (bash/read/MCP frames capped before full buffering); context budget; abort on Ctrl+C |
| Background logs | Written under the user's own data dir; command text is never interpolated into the redirect wrapper (script file + controlled argv); pruned after 7 days |
| MCP | Annotations (`readOnlyHint`) are **never** used for plan-mode authorization; all MCP tools are treated as mutating. Stdio/SSE/HTTP frames are size-capped |
| `ap serve` | Always requires a bearer token. Session `cwd` / `allowOutside` / `system` are rejected over HTTP |
| Supply chain | Zero runtime dependencies (Bun built-ins only); npm releases are published from GitHub Actions with provenance attestation; release attaches `checksums.txt` and installers verify SHA-256 when present |

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting**:
<https://github.com/Sanjay-doppalapudi/Agent-Platform/security/advisories/new>

Do not open public issues for security reports. You can expect an initial response
within a week. The latest released version is the only supported version.
