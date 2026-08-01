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

echo "installed: $dir/ap"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "note: add $dir to your PATH" ;;
esac
command -v rg >/dev/null || echo "note: ripgrep not found — install it (brew install ripgrep / apt install ripgrep)"
echo "next: ap auth <provider>   then: ap"
