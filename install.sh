#!/usr/bin/env bash
# Agent Platform (ap) installer — Linux/macOS. Downloads the latest release binary.
#   curl -fsSL https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.sh | bash
set -euo pipefail

repo="Sanjay-doppalapudi/Agent-Platform"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "$os-$arch" in
  linux-x86_64)  asset="ap-linux-x64" ;;
  darwin-arm64)  asset="ap-darwin-arm64" ;;
  darwin-x86_64) asset="ap-darwin-x64" ;;
  *) echo "unsupported platform: $os-$arch" >&2; exit 1 ;;
esac

dir="${AP_INSTALL:-$HOME/.local/bin}"
mkdir -p "$dir"
echo "downloading $asset..."
curl -fsSL "https://github.com/$repo/releases/latest/download/$asset" -o "$dir/ap"
chmod +x "$dir/ap"

if command -v sha256sum >/dev/null; then
  hash="$(sha256sum "$dir/ap" | awk '{print $1}')"
elif command -v shasum >/dev/null; then
  hash="$(shasum -a 256 "$dir/ap" | awk '{print $1}')"
else
  hash=""
fi
if [ -n "$hash" ]; then
  echo "sha256: $hash"
  if sums="$(curl -fsSL "https://github.com/$repo/releases/latest/download/checksums.txt" 2>/dev/null || true)" \
     && [ -n "$sums" ]; then
    want="$(printf '%s\n' "$sums" | awk -v a="$asset" 'tolower($2)==tolower(a) {print tolower($1); exit}')"
    if [ -n "$want" ]; then
      if [ "$want" != "$hash" ]; then
        echo "checksum mismatch for $asset (got $hash, want $want)" >&2
        rm -f "$dir/ap"
        exit 1
      fi
      echo "checksum verified against release checksums.txt"
    else
      echo "note: $asset not listed in checksums.txt — verify manually before first use"
    fi
  else
    echo "note: checksums.txt not available yet — verify the sha256 above against the GitHub release"
  fi
fi

echo "installed: $dir/ap"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "note: add $dir to your PATH" ;;
esac
command -v rg >/dev/null || echo "note: ripgrep not found — install it (brew install ripgrep / apt install ripgrep)"
echo "next: ap auth <provider>   then: ap"
