const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("hudDesktop", {
  resize: (height) => ipcRenderer.send("hud:resize", height),
  close: () => ipcRenderer.send("hud:close")
});
