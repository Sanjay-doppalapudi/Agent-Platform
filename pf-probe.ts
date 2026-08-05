import { permissionFor } from "./src/tools/index.ts";
const cfg: any = { permission: { bash: { "git push*": "ask", "rm *": "deny", "*": "allow" } } };
const cases: [string, string][] = [
  ["trailing comma", '{"cmd": "rm -rf build",}'],
  ["fenced", "```json\n{\"cmd\": \"rm -rf build\"}\n```"],
  ["double-encoded", JSON.stringify('{"cmd": "rm -rf build"}')],
  ["smart quotes", '{“cmd”: “rm -rf build”}'],
];
for (const [label, c] of cases) console.log(label.padEnd(16), "=>", permissionFor(cfg, "bash", c));
console.log("strict json".padEnd(16), "=>", permissionFor(cfg, "bash", '{"cmd":"rm -rf build"}'));

// deny-only config, no "*" fallback
const cfg2: any = { permission: { bash: { "rm *": "deny" } } };
console.log("\nno-* config, rm  =>", permissionFor(cfg2, "bash", '{"cmd":"rm -rf build"}'));
console.log("no-* config, bad =>", permissionFor(cfg2, "bash", '{"cmd": "rm -rf build",}'));
