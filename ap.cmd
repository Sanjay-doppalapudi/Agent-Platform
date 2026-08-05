@echo off
REM "ap" shim - runs AP from source via bun.
REM Smart App Control (enforcing on this machine) blocks locally compiled
REM unsigned binaries, so a freshly built ap.exe cannot launch. bun is
REM signed and runs fine, so this keeps the "ap" command working with no
REM policy changes. Costs ~150ms extra startup; behaviour is identical.
bun "%~dp0src\index.ts" %*
