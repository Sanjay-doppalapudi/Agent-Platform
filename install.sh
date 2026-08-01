#!/usr/bin/env bash
# harness installer (Linux/macOS) — downloads the latest release binary.
#   curl -fsSL https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.sh | bash
set -euo pipefail

repo="Sanjay-doppalapudi/Agent-Platform"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "$os-$arch" in
  linux-x86_64)  asset="harness-linux-x64" ;;
  darwin-arm64)  asset="harness-darwin-arm64" ;;
  darwin-x86_64) asset="harness-darwin-x64" ;;
  *) echo "unsupported platform: $os-$arch" >&2; exit 1 ;;
esac

dir="${HARNESS_INSTALL:-$HOME/.local/bin}"
mkdir -p "$dir"
echo "downloading $asset..."
curl -fsSL "https://github.com/$repo/releases/latest/download/$asset" -o "$dir/harness"
chmod +x "$dir/harness"

echo "installed: $dir/harness"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "note: add $dir to your PATH" ;;
esac
command -v rg >/dev/null || echo "note: ripgrep not found — install it (brew install ripgrep / apt install ripgrep)"
echo "next: harness auth <provider>   then: harness"
