'use strict';
const {app, BrowserWindow, ipcMain, dialog, session} = require('electron');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {normalizeEvent, validateInputAudio} = require('./protocol');
const {createDiagnostics} = require('./diagnostics');

const rendererPath = path.join(__dirname, 'renderer', 'index.html');
const rendererUrl = pathToFileURL(rendererPath).href;
function findWorkspace() {
  if (process.env.VOICE_LAB_WORKSPACE) return path.resolve(process.env.VOICE_LAB_WORKSPACE);
  // Setup writes dependencies, models, runtime, and the local app beside the
  // repository. Discovery must also work before those directories exist.
  const exampleRoot = path.dirname(__dirname);
  if (path.basename(__dirname) === 'desktop' && path.basename(exampleRoot) === 'windows_voicechat' && path.basename(path.dirname(exampleRoot)) === 'examples') {
    return path.resolve(__dirname, '../../../..');
  }
  if (path.basename(__dirname) === 'app' && path.basename(path.dirname(__dirname)) === 'resources') {
    return path.resolve(__dirname, '../../..');
  }
  return path.dirname(app.getPath('exe'));
}
const workspace = findWorkspace();
function resolveConfigPath(value) {
  return typeof value === 'string' && value.length > 0 && !value.includes('\0') ? path.resolve(workspace, value) : undefined;
}
function portableConfigPath(value) {
  const relative = path.relative(workspace, value);
  return relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative) ? relative : value;
}
const diagnostics = createDiagnostics(path.join(workspace, 'logs'), fs);
const configPath = path.join(workspace, 'voice-lab-config.json');
let config = {
  bundlePath: path.join(workspace, 'models', 'nemotron-voicechat-rtx.bundle'),
  bridgePath: path.join(workspace, 'runtime', 'trtmc_voicechat_bridge.exe'),
  systemPrompt: 'You are a warm, curious assistant speaking with someone on a live stream. Keep replies conversational and concise.',
};
try {
  const saved = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  for (const key of ['bundlePath', 'bridgePath']) {
    const resolved = resolveConfigPath(saved?.[key]);
    if (resolved) config[key] = resolved;
  }
  if (typeof saved?.systemPrompt === 'string') config.systemPrompt = saved.systemPrompt;
  // DLL search directories are an installation setting, never renderer options.
  if (Array.isArray(saved?.dllPaths) && saved.dllPaths.every(value => resolveConfigPath(value))) config.dllPaths = saved.dllPaths.map(resolveConfigPath);
} catch {}
let window, child, stopping, status = 'disconnected', gpuBusy = false;
let connectionGeneration = 0;
const emit = event => {
  diagnostics.record(event);
  if (event.type === 'state') status = event.state;
  if (window && !window.isDestroyed()) window.webContents.send('voice:event', event);
};
function assertSender(event) {
  if (!window || event.sender !== window.webContents || event.senderFrame !== window.webContents.mainFrame || event.senderFrame?.url !== rendererUrl) throw new Error('Untrusted sender');
}
function saveConfig(next = config) {
  const saved = {...next, bundlePath: portableConfigPath(next.bundlePath), bridgePath: portableConfigPath(next.bridgePath)};
  if (next.dllPaths) saved.dllPaths = next.dllPaths.map(portableConfigPath);
  fs.writeFileSync(configPath, JSON.stringify(saved, null, 2) + '\n');
}
function send(packet) {
  if (!child || child.stdin.destroyed || stopping) return false;
  if (child.stdin.writableLength > 512 * 1024) {
    emit({type: 'flush'});
    emit({type: 'error', fatal: true, message: 'The inference process cannot keep up with microphone input. Session stopped to prevent delayed audio.'});
    void disconnect();
    return false;
  }
  return child.stdin.write(JSON.stringify(packet) + '\n');
}
async function stopChild() {
  if (stopping) return stopping;
  if (!child) { emit({type: 'state', state: 'disconnected'}); return; }
  const active = child;
  stopping = new Promise(resolve => {
    const timeout = setTimeout(() => active.kill(), 5000);
    active.once('close', () => { clearTimeout(timeout); resolve(); });
    if (!active.stdin.destroyed) active.stdin.end('{"type":"stop"}\n');
    else active.kill();
  });
  await stopping;
  stopping = null;
}
async function disconnect() {
  // Invalidate a connect request that is still waiting for an old process to exit.
  connectionGeneration += 1;
  return stopChild();
}
function connectionConfig(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Invalid connection settings');
  const next = {...config};
  for (const key of ['bundlePath', 'bridgePath', 'systemPrompt']) {
    if (options[key] !== undefined) {
      if (typeof options[key] !== 'string' || options[key].includes('\0')) throw new Error(`Invalid ${key} setting.`);
      next[key] = options[key];
    }
  }
  for (const [key, label] of [['bundlePath', 'TensorRT-RTX VoiceChat bundle'], ['bridgePath', 'native Windows voice bridge']]) {
    if (!path.isAbsolute(next[key]) || !fs.statSync(next[key], {throwIfNoEntry: false})?.isFile()) throw new Error(`Choose an existing ${label}.`);
  }
  if (path.extname(next.bridgePath).toLowerCase() !== '.exe') throw new Error('The native bridge must be a Windows executable.');
  if (next.systemPrompt.length > 8192) throw new Error('System prompt is too long.');
  return next;
}
async function connect(options) {
  // Validate before altering either a running session or its persisted settings.
  const next = connectionConfig(options);
  const requestedGeneration = ++connectionGeneration;
  await stopChild();
  if (requestedGeneration !== connectionGeneration) return {state: 'disconnected', cancelled: true};
  saveConfig(next);
  config = next;
  let buffer = '', stderrTail = '', latestEpoch = -1, latestSequence = -1;
  const runtimeRoot = path.dirname(next.bridgePath);
  const runtimeCache = path.join(workspace, 'models', 'voicechat.rtx.cache');
  const args = ['--bundle', next.bundlePath, '--runtime-root', runtimeRoot, '--system-prompt', next.systemPrompt, '--runtime-cache', runtimeCache];
  const dllPaths = next.dllPaths || [];
  const active = spawn(next.bridgePath, args, {
    cwd: runtimeRoot, windowsHide: true, shell: false, stdio: ['pipe', 'pipe', 'pipe'],
    env: {...process.env, PATH: [runtimeRoot, ...dllPaths, process.env.PATH || ''].join(path.delimiter)},
  });
  child = active;
  emit({type: 'state', state: 'loading', message: 'Loading Nemotron on TensorRT-RTX…'});
  active.stdout.setEncoding('utf8');
  active.stderr.setEncoding('utf8');
  active.stdout.on('data', data => {
    if (child !== active || stopping) return;
    buffer += data;
    if (buffer.length > 12 * 1024 * 1024) { emit({type: 'flush'}); emit({type: 'error', fatal: true, message: 'Native protocol packet exceeded limit.'}); void disconnect(); return; }
    let newline;
    while ((newline = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newline).trim(); buffer = buffer.slice(newline + 1);
      if (!line) continue;
      try {
        const packet = JSON.parse(line);
        if (packet.type === 'event') {
          if (!Number.isSafeInteger(packet.epoch) || packet.epoch < 0 || !Number.isSafeInteger(packet.sequence) || packet.sequence < 0) throw new Error('Invalid native event identity');
          // Barge-in/reset advance the epoch. Older queued output cannot reappear.
          if (packet.epoch < latestEpoch || (packet.epoch === latestEpoch && packet.sequence <= latestSequence)) continue;
          latestEpoch = packet.epoch;
          latestSequence = packet.sequence;
        }
        for (const event of normalizeEvent(packet)) {
          emit(event);
          if (event.type === 'error' && event.fatal !== false) { void disconnect(); return; }
        }
      }
      catch (error) { emit({type: 'flush'}); emit({type: 'error', fatal: true, message: `Native protocol error: ${error.message}`}); void disconnect(); break; }
    }
  });
  active.stderr.on('data', data => { stderrTail = (stderrTail + data).slice(-4000); });
  active.stdin.on('error', error => {
    if (child !== active || stopping) return;
    emit({type: 'flush'}); emit({type: 'error', fatal: true, message: error.message}); void disconnect();
  });
  active.on('error', error => {
    if (child !== active || stopping) return;
    emit({type: 'flush'}); emit({type: 'error', fatal: true, message: `Cannot start voice runtime: ${error.message}`}); void disconnect();
  });
  active.on('close', (code, signal) => {
    if (child !== active) return;
    child = null;
    emit({type: 'flush'});
    if ((code || signal) && !stopping) emit({type: 'error', fatal: true, message: `Voice runtime exited (${signal || code}). ${stderrTail.trim()}`});
    emit({type: 'state', state: 'disconnected', message: signal ? 'Session stopped.' : undefined});
  });
  return {backend: 'trt_rtx', state: 'loading'};
}
function pollGpu() {
  if (gpuBusy) return;
  gpuBusy = true;
  const process = spawn('nvidia-smi', ['--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], {windowsHide: true});
  let output = '';
  const timeout = setTimeout(() => process.kill(), 3000);
  process.stdout.on('data', data => { output += data; });
  process.on('error', () => {});
  process.on('close', code => {
    clearTimeout(timeout); gpuBusy = false;
    if (code === 0) {
      const [gpuName, used, total, utilization] = output.trim().split('\n')[0].split(',').map(x => x.trim());
      emit({type: 'metrics', gpuName, gpuMemoryMb: Number(used), gpuTotalMemoryMb: Number(total), gpuUtilization: Number(utilization)});
    }
  });
}
app.setName('Nemotron Voice Lab');
// Keep portable app state with the workspace, including Chromium's cache.
app.setPath('userData', path.join(workspace, '.voice-lab'));
app.whenReady().then(() => {
  const isAudioPage = (contents, details = {}) => contents === window?.webContents && contents.getURL() === rendererUrl && details.isMainFrame !== false && (!details.requestingUrl || details.requestingUrl === rendererUrl);
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(isAudioPage(contents, details) && permission === 'media' && Array.isArray(details?.mediaTypes) && details.mediaTypes.length > 0 && details.mediaTypes.every(type => type === 'audio'));
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission, _origin, details) => isAudioPage(contents, details) && permission === 'media' && details?.mediaType === 'audio');
  window = new BrowserWindow({width: 1600, height: 960, minWidth: 1080, minHeight: 720, backgroundColor: '#090d0b', title: 'Nemotron Voice Lab', autoHideMenuBar: true,
    webPreferences: {preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true}});
  window.webContents.setWindowOpenHandler(() => ({action: 'deny'}));
  window.webContents.on('will-navigate', (event, url) => { if (url !== rendererUrl) event.preventDefault(); });
  window.loadFile(rendererPath);
  window.webContents.on('did-finish-load', pollGpu);
  const timer = setInterval(pollGpu, 2000);
  window.on('closed', () => { clearInterval(timer); window = null; void disconnect(); });
  for (const [channel, handler] of Object.entries({
    config: () => config,
    status: () => ({state: status, backend: 'trt_rtx'}),
    'choose-bundle': async () => { const result = await dialog.showOpenDialog(window, {title: 'Select a TensorRT-RTX VoiceChat bundle', filters: [{name: 'TRTMC bundle', extensions: ['bundle']}], properties: ['openFile']}); if (result.canceled) return null; config.bundlePath = result.filePaths[0]; saveConfig(); return config.bundlePath; },
    'choose-bridge': async () => { const result = await dialog.showOpenDialog(window, {title: 'Select the native voice bridge', filters: [{name: 'Windows executable', extensions: ['exe']}], properties: ['openFile']}); if (result.canceled) return null; config.bridgePath = result.filePaths[0]; saveConfig(); return config.bridgePath; },
    connect,
    disconnect,
    interrupt: () => {
      const active = child;
      if (!active || stopping || active.stdin.destroyed || status === 'loading' || status === 'disconnected') return {accepted: false};
      // write() false means buffered backpressure, not rejection of this command.
      diagnostics.record({type: 'interrupt_requested'});
      send({type: 'interrupt'});
      return {accepted: child === active && !stopping && !active.stdin.destroyed};
    },
    reset: () => send({type: 'reset'}),
    fullscreen: value => window.setFullScreen(Boolean(value)),
  })) ipcMain.handle(`voice:${channel}`, (event, ...args) => { assertSender(event); return handler(...args); });
  ipcMain.on('voice:audio', (event, packet) => {
    try { assertSender(event); if (child && status !== 'loading' && status !== 'disconnected') send(validateInputAudio(packet)); }
    catch (error) { emit({type: 'error', message: error.message}); }
  });
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => { if (child) { event.preventDefault(); disconnect().then(() => app.quit()); } });
