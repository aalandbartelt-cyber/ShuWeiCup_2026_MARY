# AI Ops HUD — Desktop Widget

Electron wrapper for `tools/agent_hud`:

- frameless, transparent, always-on-top;
- drag the HUD bar to move it;
- 4 agent icons in quota rings; hover/click opens a compact detail card;
- starts the WSL HUD server automatically if `127.0.0.1:8765` is not already running;
- closing the widget stops the server only if the widget started it;
- position is remembered in Electron `userData`.

## Install / run (Windows)

**Recommended** (avoids the UNC-path Electron unpack problem):

```powershell
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu-22.04\home\mary\shuwei2026\tools\agent_hud\desktop\install_windows.ps1
cd $env:LOCALAPPDATA\ShuWeiHUD
npm start
```

The script copies `main.js`, `preload.js`, `package.json` to
`%LOCALAPPDATA%\ShuWeiHUD` and installs Electron there with the npmmirror
binary mirror. `node_modules` is never written into the WSL share.

If you prefer the WSL path directly, `npm start` may fail with
`Electron failed to install correctly` because Electron's postinstall cannot
unpack from `\\wsl.localhost\...`; use the script instead.

If `wsl.exe` distro is not `Ubuntu-22.04`, set:

```powershell
$env:WSL_DISTRO = "YourDistro"
npm start
```

The widget reads the WSL server; no admin rights, no external APIs, no credentials.

## Build installer (optional)

Use `electron-builder` or `electron-forge` if you want a packaged `.exe`; not required for the contest. The current `npm start` is enough for daily use.
