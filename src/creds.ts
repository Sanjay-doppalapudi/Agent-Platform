// Separate credential store: ~/.harness/credentials.json — keys live apart
// from shareable config, file locked to the current user (icacls / chmod 600).
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

let cache: Record<string, string> | null = null;
let cachePath = "";

function credFile(dataDir: string): string {
  return join(dataDir, "credentials.json");
}

function load(dataDir: string): Record<string, string> {
  const p = credFile(dataDir);
  if (cache && cachePath === p) return cache;
  cachePath = p;
  try {
    cache = existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : {};
  } catch {
    cache = {};
  }
  return cache!;
}

export function getKey(dataDir: string, provider: string): string | undefined {
  return load(dataDir)[provider];
}

export function listKeys(dataDir: string): string[] {
  return Object.keys(load(dataDir));
}

export function setKey(dataDir: string, provider: string, key: string) {
  const p = credFile(dataDir);
  const obj = { ...load(dataDir), [provider]: key };
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(obj, null, 2) + "\n");
  lockDown(p);
  cache = obj;
}

/** Restrict the file to the current user (best-effort). */
function lockDown(p: string) {
  try {
    if (process.platform === "win32") {
      // Fully-qualified DOMAIN\user — a bare name is ambiguous when the
      // machine name matches the username and icacls then grants to nobody.
      const user = process.env.USERNAME || process.env.USER || "";
      const domain = process.env.USERDOMAIN || "";
      const account = domain ? `${domain}\\${user}` : user;
      if (user) Bun.spawnSync(["icacls", p, "/inheritance:r", "/grant:r", `${account}:F`], { stdout: "ignore", stderr: "ignore" });
    } else {
      chmodSync(p, 0o600);
    }
  } catch {}
}
