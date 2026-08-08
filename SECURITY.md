# Security Policy

## What AP is (threat model)

AP is a **local development tool** that executes commands, edits files, and fetches
URLs chosen by an LLM — inside a session that the local user starts, watches, and
can abort at any moment. It is *dual-use by design*: the same capability that fixes
your build can delete a file. AP is **not a privilege boundary**: it runs with the
invoking user's permissions, and its guardrails are engineering controls against
*accidents and model misbehavior*, not a substitute for OS-level isolation.

**Run genuinely untrusted code or goals in a container/VM.** The sandbox here is a
guardrail, not a VM — pattern scanning is best-effort, and network egress is not
fully restricted (cloud-metadata hosts are blocked in `fetch`; RFC1918 / localhost
remain reachable for local docs).

## Guardrails (defense in depth)

| Layer | Mechanism |
|---|---|
| Write sandbox | `write`/`edit`/`bash` mutations outside the workspace (plus `dataDir/memory` and `dataDir/artifacts`) require an interactive user permit (`y/N/always`); headless runs auto-deny unless `--allow-outside`. The rest of `<dataDir>` is **not** freely writable |
| Read scoping | Reads outside the workspace + skills/memory/commands dirs also require a permit; bash commands are path-token scanned (incl. `../` escapes). Symlink targets are resolved before the containment check |
| AP-private data | Session transcripts, checkpoints, `credentials.json`, `config.json` are **hard-denied for read and write — permits cannot override** |
| Dangerous commands | Destructive patterns (recursive absolute deletes, disk format, registry edits, `curl \| sh`, `bash <(…)`, PowerShell IEX/encoded, `find -delete`, fork bombs, …) are blocked outright — never prompted — and logged to `<dataDir>/blocked-commands.jsonl`. Per-tool `permission.bash` patterns apply to **each segment** of compound commands (`&&`/`||`/`|`/`;`); the strictest verdict wins |
| Plan mode | Structurally read-only: mutating tool schemas are not even sent to the model, plus a runtime backstop for hallucinated calls. `/confirm edits on` additionally forces ask on every edit/write even under yolo |
| Hooks | Hook command strings come **only from the user's own config files**, never from model output; tool arguments are passed via environment variables, never interpolated into the command line (`afterEdit` also receives `AP_ARGS.paths`) |
| Credentials | Stored in `<dataDir>/credentials.json`, file-ACL'd to the invoking user; `.env` / common secret filenames are redacted in `read` **and** `grep` output by default |
| `ap serve` | Binds `127.0.0.1` by default (not `0.0.0.0`). Non-loopback binds require a bearer token (`--token` / `AP_SERVE_TOKEN`); `/health` does not advertise provider/model |
| Fetch | Cloud metadata / link-local hosts (`169.254.0.0/16`, `metadata.google.internal`, `fd00:ec2::254`, …) are refused — including **IPv4 embedded in IPv6** (`::ffff:…`). DNS is resolved (`lookup` `{all:true}`) and the connection pinned to a vetted address; every redirect is validated before connecting. Localhost / RFC1918 stay allowed for local docs |
| Artifacts | Model HTML gets a no-network CSP (`default-src 'none'`); slug path segments are alphabet-validated and containment-asserted |
| Real git | `/commit` and `/pr` never push or force-push; protected branches refuse commits unless the user switches (or `git.autoBranch` creates `ap/<slug>`). `ap pr` requires `--yes` / `AP_PR_YES=1` to create |
| Resource limits | Per-command timeouts with process-tree kill, output caps on every tool (30–50KB), context budget, abort on Ctrl+C |
| Background logs | Written under the user's own data dir; pruned automatically after 7 days |
| Supply chain | Zero runtime dependencies (Bun built-ins only); npm releases are published from GitHub Actions with provenance attestation |

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting**:
<https://github.com/Sanjay-doppalapudi/Agent-Platform/security/advisories/new>

Do not open public issues for security reports. You can expect an initial response
within a week. The latest released version is the only supported version.
