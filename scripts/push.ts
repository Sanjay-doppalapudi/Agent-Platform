// Push wrapper — THE way to push this repo (rule: every push to GitHub bumps
// the version by 0.0.1). Bumps package.json + the CLI VERSION constant,
// commits, tags v<version>, and pushes with tags (which also triggers the
// release workflow so the binaries stay current).
//
//   bun run push                 → commit message "v0.0.X"
//   bun run push "my message"    → custom message, still tagged v0.0.X
import { readFileSync, writeFileSync } from "node:fs";

const run = (cmd: string[]) => {
  const p = Bun.spawnSync(cmd, { stdout: "inherit", stderr: "inherit" });
  if (p.exitCode !== 0) {
    console.error(`failed: ${cmd.join(" ")}`);
    process.exit(p.exitCode ?? 1);
  }
};

const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const parts = String(pkg.version).split(".").map(Number);
const next = `${parts[0]}.${parts[1]}.${(parts[2] ?? 0) + 1}`;
pkg.version = next;
writeFileSync("package.json", JSON.stringify(pkg, null, 2) + "\n");

const idx = readFileSync("src/index.ts", "utf8");
writeFileSync("src/index.ts", idx.replace(/const VERSION = "[^"]+"/, `const VERSION = "${next}"`));

const msg = process.argv[2] ?? `v${next}`;
run(["git", "add", "-A"]);
run(["git", "commit", "-m", msg]);
run(["git", "tag", `v${next}`]);
run(["git", "push", "origin", "main", "--tags"]);
console.log(`\npushed v${next}`);
