'use strict';
const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('voiceLab', {
  getConfig: () => ipcRenderer.invoke('voice:config'),
  getStatus: () => ipcRenderer.invoke('voice:status'),
  chooseBundle: () => ipcRenderer.invoke('voice:choose-bundle'),
  chooseBridge: () => ipcRenderer.invoke('voice:choose-bridge'),
  connect: config => ipcRenderer.invoke('voice:connect', config),
  disconnect: () => ipcRenderer.invoke('voice:disconnect'),
  sendAudio: packet => ipcRenderer.send('voice:audio', packet),
  reset: () => ipcRenderer.invoke('voice:reset'),
  interrupt: () => ipcRenderer.invoke('voice:interrupt'),
  setFullscreen: value => ipcRenderer.invoke('voice:fullscreen', value),
  onEvent: callback => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on('voice:event', handler);
    return () => ipcRenderer.removeListener('voice:event', handler);
  },
});
