# harness installer (Windows) — downloads the latest release binary.
#   irm https://raw.githubusercontent.com/Sanjay-doppalapudi/Agent-Platform/main/install.ps1 | iex
$ErrorActionPreference = "Stop"
$repo = "Sanjay-doppalapudi/Agent-Platform"
$asset = "harness-windows-x64.exe"
$dir = "$env:LOCALAPPDATA\harness"

New-Item -ItemType Directory -Force $dir | Out-Null
Write-Host "downloading $asset..."
Invoke-WebRequest "https://github.com/$repo/releases/latest/download/$asset" -OutFile "$dir\harness.exe"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $dir) {
  [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $dir), "User")
  Write-Host "added $dir to user PATH (open a new terminal)"
}

Write-Host "installed: $dir\harness.exe"
if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
  Write-Host "note: ripgrep not found — install it with: winget install BurntSushi.ripgrep.MSVC"
}
Write-Host "next: harness auth <provider>   then: harness"
