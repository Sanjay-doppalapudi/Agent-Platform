# Security Policy

## What AP is (threat model)

AP is a **local development tool** that executes commands, edits files, and fetches
URLs chosen by an LLM — inside a session that the local user starts, watches, and
can abort at any moment. It is *dual-use by design*: the same capability that fixes
your build can delete a file. AP is **not a privilege boundary**: it runs with the
invoking user's permissions, and its guardrails are engineering controls against
*accidents and model misbehavior*, not a substitute for OS-level isolation.

**Run genuinely untrusted code or goals in a container/VM.** The sandbox here is a
guardrail, not a VM — pattern scanning is best-effort, symlinks are not resolved,
and network egress is not restricted.

## Guardrails (defense in depth)

| Layer | Mechanism |
|---|---|
| Write sandbox | `write`/`edit`/`bash` mutations outside the workspace require an interactive user permit (`y/N/always`); headless runs auto-deny unless `--allow-outside` |
| Read scoping | Reads outside the workspace + skills/memory dirs also require a permit; bash commands are path-token scanned (incl. `../` escapes) |
| AP-private data | Session transcripts, checkpoints, `credentials.json`, `config.json` are **hard-denied for read and write — permits cannot override** |
| Dangerous commands | Destructive patterns (recursive absolute deletes, disk format, registry edits, `curl \| sh`, fork bombs, …) are blocked outright — never prompted — and logged to `<dataDir>/blocked-commands.jsonl` |
| Plan mode | Structurally read-only: mutating tool schemas are not even sent to the model, plus a runtime backstop for hallucinated calls |
| Hooks | Hook command strings come **only from the user's own config files**, never from model output; tool arguments are passed via environment variables, never interpolated into the command line |
| Credentials | Stored in `<dataDir>/credentials.json`, file-ACL'd to the invoking user; `.env` values are redacted (`KEY=***`) in tool output by default |
| Resource limits | Per-command timeouts with process-tree kill, output caps on every tool (30–50KB), context budget, abort on Ctrl+C |
| Background logs | Written under the user's own data dir; pruned automatically after 7 days |
| Supply chain | Zero runtime dependencies (Bun built-ins only); npm releases are published from GitHub Actions with provenance attestation |

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting**:
<https://github.com/Sanjay-doppalapudi/Agent-Platform/security/advisories/new>

Do not open public issues for security reports. You can expect an initial response
within a week. The latest released version is the only supported version.
