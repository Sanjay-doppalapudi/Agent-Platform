// Example AP workflow. Copy to .ap/workflows/review.ts in your repo, then:
//   ap flow review "src/foo.ts src/bar.ts"        (or /flow review … in the REPL)
//
// It fans a review across the given files in parallel, then adversarially
// verifies each finding with a second agent — the find→refute pattern, but
// tiny. Every agent() call is a --light subagent; `parallel` caps concurrency.
export default async function ({ agent, parallel, log, args }) {
  const files = (args.join(" ").trim() || "the files changed in the last commit")
    .split(/\s+/)
    .filter(Boolean);
  log(`reviewing ${files.length} target(s)`);

  const FINDINGS = {
    type: "object",
    properties: {
      findings: {
        type: "array",
        items: {
          type: "object",
          properties: { title: { type: "string" }, detail: { type: "string" } },
          required: ["title", "detail"],
        },
      },
    },
    required: ["findings"],
  };

  // Stage 1: one reviewer per file, in parallel.
  const reviews = await parallel(
    files.map((f) => () =>
      agent(`Review ${f} for real bugs (correctness, resource leaks, edge cases). Read it first.`, { schema: FINDINGS })),
  );
  const findings = reviews
    .filter(Boolean)
    .flatMap((r, i) => (r.findings ?? []).map((x) => ({ ...x, file: files[i] })));
  log(`${findings.length} candidate findings`);

  // Stage 2: refute each one — keep only what survives.
  const verdicts = await parallel(
    findings.map((f) => () =>
      agent(`In ${f.file}, is this a REAL bug? "${f.title}: ${f.detail}". Read the code and answer CONFIRMED or REFUTED with one line of reasoning.`)),
  );

  return findings
    .map((f, i) => ({ ...f, verdict: verdicts[i] ?? "(no verdict)" }))
    .filter((f) => /CONFIRMED/i.test(String(f.verdict)));
}
