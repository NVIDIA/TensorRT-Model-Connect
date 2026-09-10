'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {EventEmitter} = require('node:events');
const {PassThrough} = require('node:stream');
const {pathToFileURL} = require('node:url');
const mainPath = path.resolve(__dirname, '..', 'main.js');
const source = fs.readFileSync(mainPath, 'utf8');
const workspace = path.resolve(__dirname, 'fake-workspace');
const bridge = path.join(workspace, 'runtime', 'bridge.exe');
const bundle = path.join(workspace, 'models', 'voice.bundle');
const ready = {type: 'ready', backend: 'trt_rtx', family: 'nemotron_voicechat', protocolVersion: 1, inputSampleRate: 16000, outputSampleRate: 48000};

async function harness(t, saved = {}, options = {}) {
  const messages = [], children = [], writes = [], handlers = new Map();
  const appDirectory = options.appDirectory || path.dirname(mainPath);
  const app = new EventEmitter();
  app.whenReady = () => Promise.resolve();
  app.getPath = () => options.exePath || path.join(workspace, 'VoiceLab.exe');
  app.setName = app.setPath = app.quit = () => {};
  const session = {defaultSession: {
    setPermissionRequestHandler(handler) { this.request = handler; },
    setPermissionCheckHandler(handler) { this.check = handler; },
  }};
  let window;
  class BrowserWindow extends EventEmitter {
    constructor() {
      super(); window = this;
      const contents = new EventEmitter();
      contents.mainFrame = {url: pathToFileURL(path.join(appDirectory, 'renderer', 'index.html')).href};
      contents.getURL = () => contents.mainFrame.url;
      contents.send = (_channel, event) => messages.push(event);
      contents.setWindowOpenHandler = () => {};
      this.webContents = contents;
    }
    isDestroyed() { return false; }
    loadFile() {}
  }
  const ipcMain = new EventEmitter();
  ipcMain.handle = (channel, handler) => handlers.set(channel, handler);
  const fakeFs = {
    readFileSync: () => JSON.stringify(options.emptyConfig ? saved : {bundlePath: bundle, bridgePath: bridge, ...saved}),
    writeFileSync: (_file, content) => writes.push(JSON.parse(content)),
    statSync: file => [bundle, bridge, ...(options.files || [])].includes(file) ? {isFile: () => true} : undefined,
    existsSync: () => false,
  };
  const spawn = (file, args, options) => {
    const child = new EventEmitter();
    child.file = file; child.args = args; child.options = options;
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough();
    child.kill = () => { child.emit('close', null, 'SIGTERM'); return true; };
    child.packet = packet => child.stdout.write(JSON.stringify(packet) + '\n');
    children.push(child);
    return child;
  };
  const timers = new Set();
  const sandbox = {
    __dirname: appDirectory, Buffer, console,
    process: {env: options.env || {VOICE_LAB_WORKSPACE: workspace, PATH: 'system-path'}},
    setTimeout(fn, delay) { const timer = setTimeout(fn, delay); timers.add(timer); return timer; },
    clearTimeout(timer) { clearTimeout(timer); timers.delete(timer); },
    setInterval: () => 0, clearInterval: () => {},
    require(name) {
      if (name === 'electron') return {app, BrowserWindow, ipcMain, session, dialog: {}};
      if (name === 'node:child_process') return {spawn};
      if (name === 'node:fs') return fakeFs;
      if (name === './protocol') return require('../protocol');
      if (name === './diagnostics') return require('../diagnostics');
      return require(name);
    },
  };
  vm.runInNewContext(source, sandbox, {filename: mainPath});
  await Promise.resolve();
  t.after(() => { for (const timer of timers) clearTimeout(timer); for (const child of children) child.emit('close', 0, null); });
  const sender = {sender: window.webContents, senderFrame: window.webContents.mainFrame};
  const invoke = (name, ...args) => handlers.get(`voice:${name}`)(sender, ...args);
  return {invoke, messages, children, writes, session: session.defaultSession, window, handlers, sender};
}

test('a source checkout discovers its workspace before dependencies and models exist', async t => {
  const root = path.join(workspace, 'new-install');
  const h = await harness(t, {}, {emptyConfig: true, env: {}, appDirectory: path.join(root, 'TensorRT-Model-Connect', 'examples', 'windows_voicechat', 'desktop')});
  assert.equal(h.invoke('config').bundlePath, path.join(root, 'models', 'nemotron-voicechat-rtx.bundle'));
  assert.equal(h.invoke('config').bridgePath, path.join(root, 'runtime', 'trtmc_voicechat_bridge.exe'));
});

test('a relocated local app resolves saved paths and launches from its new workspace', async t => {
  const root = path.join(workspace, 'moved-install');
  const movedBridge = path.join(root, 'runtime', 'bridge.exe');
  const movedBundle = path.join(root, 'models', 'voice.bundle');
  const h = await harness(t, {
    bridgePath: path.join('runtime', 'bridge.exe'), bundlePath: path.join('models', 'voice.bundle'), dllPaths: [path.join('dependencies', 'cuda')],
  }, {env: {}, appDirectory: path.join(root, 'Nemotron Voice Lab', 'resources', 'app'), files: [movedBridge, movedBundle]});
  await h.invoke('connect', {});
  assert.equal(h.children[0].file, movedBridge);
  assert(h.children[0].args.includes(movedBundle));
  assert(h.children[0].args.includes(path.join(root, 'models', 'voicechat.rtx.cache')));
  assert(h.children[0].options.env.PATH.includes(path.join(root, 'dependencies', 'cuda')));
  assert.equal(h.writes[0].bridgePath, path.join('runtime', 'bridge.exe'));
  assert.equal(h.writes[0].bundlePath, path.join('models', 'voice.bundle'));
  assert.deepEqual(h.writes[0].dllPaths, [path.join('dependencies', 'cuda')]);
});

test('the explicit workspace wins and external model selections stay absolute', async t => {
  const externalBundle = path.resolve(workspace, '..', 'shared-models', 'voice.bundle');
  const h = await harness(t, {bundlePath: externalBundle}, {files: [externalBundle]});
  await h.invoke('connect', {});
  assert.equal(h.children[0].file, bridge);
  assert.equal(h.writes[0].bridgePath, path.join('runtime', 'bridge.exe'));
  assert.equal(h.writes[0].bundlePath, externalBundle);
});

test('invalid connection settings preserve the active process and prior config', async t => {
  const h = await harness(t);
  await h.invoke('connect', {systemPrompt: 'Working session'});
  const active = h.children[0];
  active.packet(ready);
  await assert.rejects(h.invoke('connect', {bundlePath: path.join(workspace, 'missing.bundle')}), /Choose an existing/);
  assert.equal(h.children.length, 1);
  assert.equal(active.stdin.writableEnded, false);
  assert.equal(h.invoke('config').bundlePath, bundle);
  assert.equal(h.invoke('status').state, 'listening');
  assert.equal(h.writes.length, 1);
});

test('simultaneous connects create one process and persist only the winning request', async t => {
  const h = await harness(t);
  const first = h.invoke('connect', {systemPrompt: 'First'});
  const second = h.invoke('connect', {systemPrompt: 'Second'});
  assert.equal((await first).cancelled, true);
  assert.equal((await second).state, 'loading');
  assert.equal(h.children.length, 1);
  assert.equal(h.writes.length, 1);
  assert.equal(h.invoke('config').systemPrompt, 'Second');
});

test('disconnect cancels a pending reconnect and suppresses late native readiness', async t => {
  const h = await harness(t);
  await h.invoke('connect', {});
  const old = h.children[0];
  old.packet(ready);
  const next = h.invoke('connect', {systemPrompt: 'Next'});
  assert.equal(old.stdin.writableEnded, true);
  const messageCount = h.messages.length;
  old.packet(ready);
  assert.equal(h.messages.length, messageCount, 'A stopping process cannot revive the renderer.');
  const disconnect = h.invoke('disconnect');
  old.emit('close', 0, null);
  await disconnect;
  assert.equal((await next).cancelled, true);
  assert.equal(h.children.length, 1);
  assert.equal(h.invoke('status').state, 'disconnected');
});

test('barge-in rejects stale epochs and duplicate text while retaining new output', async t => {
  const h = await harness(t);
  await h.invoke('connect', {});
  const active = h.children[0];
  active.packet(ready);
  active.packet({type: 'event', kind: 'agent_text', epoch: 2, sequence: 0, text: 'Old', isFinal: false});
  active.packet({type: 'event', kind: 'yielded', epoch: 3, sequence: 0});
  active.packet({type: 'event', kind: 'agent_text', epoch: 2, sequence: 1, text: ' stale', isFinal: false});
  active.packet({type: 'event', kind: 'agent_text', epoch: 4, sequence: 0, text: 'New', isFinal: false});
  active.packet({type: 'event', kind: 'agent_text', epoch: 4, sequence: 0, text: 'New', isFinal: false});
  assert.deepEqual(h.messages.filter(event => event.type === 'transcript').map(event => event.text), ['Old', 'New']);
  assert(h.messages.some(event => event.type === 'flush' && event.epoch === 3));
});

test('interrupt sends an independent control without stopping microphone input or the session', async t => {
  const h = await harness(t);
  assert.equal(h.invoke('interrupt').accepted, false);
  await h.invoke('connect', {});
  const active = h.children[0];
  assert.equal(h.invoke('interrupt').accepted, false, 'Do not interrupt a loading session.');
  active.packet(ready);
  const commands = [];
  active.stdin.on('data', data => commands.push(JSON.parse(data.toString())));
  const write = active.stdin.write.bind(active.stdin);
  active.stdin.write = (...args) => { write(...args); return false; };
  assert.equal(h.invoke('interrupt').accepted, true, 'Buffered writes are still accepted.');
  assert.deepEqual(commands, [{type: 'interrupt'}]);
  assert.equal(active.stdin.writableEnded, false);
  active.packet({type: 'flush', reason: 'interrupt', interruptStatus: 'already_idle'});
  assert.equal(h.invoke('status').state, 'listening');
  assert(h.messages.some(event => event.type === 'flush' && event.interruptStatus === 'already_idle'));
});

test('FIFO rollover trace preserves published speech and accepts interleaved event identities', async t => {
  const h = await harness(t);
  await h.invoke('connect', {});
  const active = h.children[0];
  active.packet(ready);
  const traceStart = h.messages.length;
  const event = (kind, epoch, sequence, fields = {}) => ({type: 'event', kind, epoch, sequence, ...fields});
  const audio = (epoch, sequence, samples) => {
    const bytes = Buffer.alloc(samples.length * 4);
    samples.forEach((sample, index) => bytes.writeFloatLE(sample, index * 4));
    return event('agent_audio', epoch, sequence, {sampleRate: 48000, encoding: 'f32le', sampleCount: samples.length, audio: bytes.toString('base64')});
  };
  // publish_current_event, publish_agent_event, and emit_audio share one
  // sequence counter. Text and audio do not have independent sequence streams.
  const beforeRollover = [
    event('user_speech_started', 1, 0),
    event('user_transcript', 1, 1, {text: 'Tell me', isFinal: false}),
    event('user_speech_stopped', 1, 2),
    event('user_transcript', 1, 3, {text: 'Tell me a story.', isFinal: true}),
    event('turn_started', 2, 0),
    event('agent_text', 2, 1, {text: 'Once', isFinal: false}),
    audio(2, 2, [.25, -.5]),
    event('agent_text', 2, 3, {text: ' upon a time.', isFinal: false}),
    audio(2, 4, [.125, .75]),
    event('agent_text', 2, 5, {text: 'Once upon a time.', isFinal: true}),
    event('turn_finished', 2, 6),
  ];
  active.stdout.write(beforeRollover.map(packet => JSON.stringify(packet)).join('\n') + '\n');
  // finish_agent_turn queues its final events before advancing to epoch 3.
  // maybe_rollover_context leaves that epoch/counter intact. Repeated silent
  // rollovers therefore increase sequence within the SAME listening epoch.
  const afterRollover = [
    event('context_rolled', 3, 0, {text: 'segment=1 reason=age', isFinal: true}),
    event('context_rolled', 3, 1, {text: 'segment=2 reason=age', isFinal: true}),
    event('user_speech_started', 3, 2),
    event('user_transcript', 3, 3, {text: 'Go', isFinal: false}),
    event('user_speech_stopped', 3, 4),
    event('user_transcript', 3, 5, {text: 'Go on.', isFinal: true}),
    event('turn_started', 4, 0),
    event('agent_text', 4, 1, {text: 'The story continues.', isFinal: false}),
    audio(4, 2, [-.125, 0]),
  ];
  active.stdout.write(afterRollover.map(packet => JSON.stringify(packet)).join('\n') + '\n');
  const delivered = h.messages.slice(traceStart);
  assert.equal(delivered.length, beforeRollover.length + afterRollover.length, 'Every legitimately ordered event must survive normalization and filtering.');
  assert.deepEqual(delivered.map(packet => [packet.epoch, packet.sequence]), [...beforeRollover, ...afterRollover].map(packet => [packet.epoch, packet.sequence]));
  assert.deepEqual(delivered.filter(packet => packet.type === 'audio').map(packet => packet.samples), [[.25, -.5], [.125, .75], [-.125, 0]]);
  assert.equal(delivered.filter(packet => packet.type === 'context_rolled').length, 2);
  assert(!delivered.some(packet => packet.type === 'flush' || packet.type === 'error'), 'A rollover must never invalidate already-published playback.');
});

test('fatal errors stop the process and flush pending speech immediately', async t => {
  const h = await harness(t);
  await h.invoke('connect', {});
  const active = h.children[0];
  active.packet(ready);
  active.packet({type: 'event', kind: 'error', text: 'GPU failure', epoch: 1, sequence: 0});
  assert.equal(active.stdin.writableEnded, true);
  const errorIndex = h.messages.findIndex(event => event.type === 'error');
  assert.equal(h.messages[errorIndex - 1].type, 'flush');
  assert.equal(h.messages[errorIndex].message, 'GPU failure');
  const count = h.messages.length;
  active.packet(ready);
  assert.equal(h.messages.length, count);
});

test('renderer options cannot alter installed DLL search directories', async t => {
  const installedPath = path.join(workspace, 'dependencies', 'cuda');
  const h = await harness(t, {dllPaths: [installedPath]});
  await h.invoke('connect', {dllPaths: [path.join(workspace, 'untrusted')], systemPrompt: 'Valid'});
  assert(h.children[0].options.env.PATH.includes(installedPath));
  assert(!h.children[0].options.env.PATH.includes('untrusted'));
  assert.deepEqual(Array.from(h.invoke('config').dllPaths), [installedPath]);
});

test('process launch failures retain their actionable error after child closure', async t => {
  const h = await harness(t);
  await h.invoke('connect', {});
  const active = h.children[0];
  active.emit('error', new Error('Executable cannot be started'));
  active.emit('close', -2, null);
  const errors = h.messages.filter(event => event.type === 'error');
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /Executable cannot be started/);
  assert.equal(h.invoke('status').state, 'disconnected');
});

test('only the main renderer frame may call IPC or request microphone access', async t => {
  const h = await harness(t);
  const url = h.sender.senderFrame.url;
  assert.throws(() => h.handlers.get('voice:config')({...h.sender, senderFrame: {url}}), /Untrusted sender/);
  assert.equal(h.session.check(h.window.webContents, 'media', 'file://', {mediaType: 'audio', isMainFrame: true, requestingUrl: url}), true);
  assert.equal(h.session.check(h.window.webContents, 'media', 'file://', {mediaType: 'video', isMainFrame: true}), false);
  assert.equal(h.session.check(h.window.webContents, 'media', 'file://', {mediaType: 'audio', isMainFrame: false}), false);
  let allowed;
  h.session.request(h.window.webContents, 'media', value => { allowed = value; }, {mediaTypes: ['audio'], isMainFrame: true});
  assert.equal(allowed, true);
  h.session.request(h.window.webContents, 'media', value => { allowed = value; }, {mediaTypes: ['audio', 'video'], isMainFrame: true});
  assert.equal(allowed, false);
});
