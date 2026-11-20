const { app, BrowserWindow, ipcMain, screen, dialog } = require("electron");
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const WIDTH = 700;
const CLOSED_HEIGHT = 112;
const OPEN_HEIGHT = 560;
const URL = "http://127.0.0.1:8765/";
const POS_FILE = () => path.join(app.getPath("userData"), "hud-position.json");

let win = null;
let serverProc = null;
let ownsServer = false;

function probe() {
  return new Promise((resolve) => {
    const req = http.get(`${URL}healthz`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(500, () => { req.destroy(); resolve(false); });
  });
}

async function ensureServer() {
  if (await probe()) return true;
  const distro = process.env.WSL_DISTRO || "Ubuntu-22.04";
  serverProc = spawn("wsl.exe", [
    "-d", distro, "-e", "bash", "-lc",
    "cd ~/shuwei2026 && python3 tools/agent_hud/server.py --port 8765"
  ], { stdio: "ignore", windowsHide: true });
  ownsServer = true;
  for (let i = 0; i < 24; i++) {
    await new Promise((r) => setTimeout(r, 250));
    if (await probe()) return true;
  }
  return false;
}

function stopServer() {
  if (!ownsServer || !serverProc) return;
  try {
    execFileSync("taskkill", ["/PID", String(serverProc.pid), "/T", "/F"], { stdio: "ignore" });
  } catch {}
  serverProc = null;
  ownsServer = false;
}

function clampPosition(pos) {
  const displays = screen.getAllDisplays();
  const cx = pos.x + WIDTH / 2;
  const cy = pos.y + CLOSED_HEIGHT / 2;
  const display = displays.find((d) => {
    const a = d.workArea;
    return cx >= a.x && cx <= a.x + a.width && cy >= a.y && cy <= a.y + a.height;
  }) || screen.getPrimaryDisplay();
  const a = display.workArea;
  return {
    x: Math.max(a.x + 8, Math.min(pos.x, a.x + a.width - WIDTH - 8)),
    y: Math.max(a.y + 8, Math.min(pos.y, a.y + a.height - CLOSED_HEIGHT - 8))
  };
}

function readPosition() {
  try {
    const p = JSON.parse(fs.readFileSync(POS_FILE(), "utf8"));
    if (Number.isFinite(p.x) && Number.isFinite(p.y)) return clampPosition(p);
  } catch {}
  const area = screen.getPrimaryDisplay().workArea;
  return clampPosition({ x: area.x + area.width - WIDTH - 20, y: area.y + 20 });
}

function savePosition() {
  if (!win) return;
  try {
    const [x, y] = win.getPosition();
    fs.writeFileSync(POS_FILE(), JSON.stringify({ x, y }));
  } catch {}
}

function createWindow() {
  const pos = readPosition();
  win = new BrowserWindow({
    width: WIDTH,
    height: CLOSED_HEIGHT,
    x: pos.x,
    y: pos.y,
    frame: false,
    transparent: true,
    backgroundColor: "#00000000",
    resizable: false,
    movable: true,
    alwaysOnTop: true,
    skipTaskbar: false,
    hasShadow: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });
  win.setMenuBarVisibility(false);
  win.setAlwaysOnTop(true);
  win.on("moved", savePosition);
  win.once("ready-to-show", () => win.show());
  win.loadURL(URL);
}

ipcMain.on("hud:resize", (_event, height) => {
  if (!win || !Number.isFinite(height)) return;
  const b = win.getBounds();
  const h = Math.max(CLOSED_HEIGHT, Math.min(OPEN_HEIGHT, Math.round(height)));
  win.setBounds({ x: b.x, y: b.y, width: WIDTH, height: h });
});

ipcMain.on("hud:close", () => app.quit());

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) { win.show(); win.focus(); }
  });
  app.whenReady().then(async () => {
    const ok = await ensureServer();
    if (!ok) {
      dialog.showErrorBox(
        "AI Ops HUD",
        "无法启动 WSL 数据服务。请确认 WSL 中有 ~/shuwei2026/tools/agent_hud/server.py，或在终端手动运行它。"
      );
      app.quit();
      return;
    }
    createWindow();
  });
}

app.on("before-quit", stopServer);
app.on("window-all-closed", () => app.quit());
