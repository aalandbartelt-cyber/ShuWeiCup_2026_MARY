# Install the Electron HUD into a Windows-local folder.
# Electron's postinstall cannot reliably unpack from a \\wsl.localhost UNC path,
# so copy the three shell files to %LOCALAPPDATA%\ShuWeiHUD and install there.

$ErrorActionPreference = "Stop"
$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst = Join-Path $env:LOCALAPPDATA "ShuWeiHUD"

New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item (Join-Path $src "main.js") $dst -Force
Copy-Item (Join-Path $src "preload.js") $dst -Force
Copy-Item (Join-Path $src "package.json") $dst -Force

$env:ELECTRON_MIRROR = "https://npmmirror.com/mirrors/electron/"
Push-Location $dst
try {
    npm install --no-audit --no-fund
    if (-not (Test-Path (Join-Path $dst "node_modules\electron\dist\electron.exe"))) {
        node (Join-Path $dst "node_modules\electron\install.js")
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Installed: $dst"
Write-Host "Start:     cd `"$dst`"; npm start"
Write-Host "The widget starts the WSL HUD server automatically if 127.0.0.1:8765 is down."
