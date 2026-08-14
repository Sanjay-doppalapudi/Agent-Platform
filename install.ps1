# Agent Platform (ap) installer — Windows. Downloads the latest release binary.
#   irm https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.ps1 | iex
$ErrorActionPreference = "Stop"
$repo = "Sanjay-doppalapudi/Agent-Platform"
$asset = "ap-windows-x64.exe"
$dir = "$env:LOCALAPPDATA\ap"

New-Item -ItemType Directory -Force $dir | Out-Null
Write-Host "downloading $asset..."
$out = "$dir\ap.exe"
Invoke-WebRequest "https://github.com/$repo/releases/latest/download/$asset" -OutFile $out

# Print SHA-256 so the user can compare against the release checksums file.
$hash = (Get-FileHash -Algorithm SHA256 $out).Hash.ToLowerInvariant()
Write-Host "sha256: $hash"
$sumsUrl = "https://github.com/$repo/releases/latest/download/checksums.txt"
try {
  $sums = (Invoke-WebRequest $sumsUrl -UseBasicParsing).Content
  if ($sums -match "(?im)^([a-f0-9]{64})\s+$([regex]::Escape($asset))\s*$") {
    $want = $Matches[1].ToLowerInvariant()
    if ($want -ne $hash) { throw "checksum mismatch for $asset (got $hash, want $want)" }
    Write-Host "checksum verified against release checksums.txt"
  } else {
    Write-Host "note: $asset not listed in checksums.txt — verify manually before first use"
  }
} catch {
  if ($_.Exception.Message -match "checksum mismatch") { throw }
  Write-Host "note: checksums.txt not available yet — verify the sha256 above against the GitHub release"
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $dir) {
  [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $dir), "User")
  Write-Host "added $dir to user PATH (open a new terminal)"
}

Write-Host "installed: $dir\ap.exe"
if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
  Write-Host "note: ripgrep not found — install it with: winget install BurntSushi.ripgrep.MSVC"
}
Write-Host "next: ap auth <provider>   then: ap"
