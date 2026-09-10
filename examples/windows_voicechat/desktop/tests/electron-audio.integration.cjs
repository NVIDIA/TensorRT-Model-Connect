'use strict';

// Opt-in integration test: ELECTRON_EXECUTABLE=/path/to/electron node this-file.
// Install Playwright locally, or point PLAYWRIGHT_MODULE at its installed package.
// The test replaces the native connection/interruption handlers in Electron main.
// The renderer, isolated preload, microphone worklet, IPC and speakers are real.
const {_electron} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

async function run() {
  const desktop = path.resolve(__dirname, '..');
  const testRoot = path.resolve(process.env.VOICE_LAB_TEST_ROOT || os.tmpdir());
  fs.mkdirSync(testRoot, {recursive: true});
  const workspace = fs.mkdtempSync(path.join(testRoot, 'voice-lab-audio-test-'));
  const env = {...process.env, VOICE_LAB_WORKSPACE: workspace};
  // Electron tests must remove this variable entirely, including an empty value.
  delete env.ELECTRON_RUN_AS_NODE;
  const executablePath = path.resolve(process.env.ELECTRON_EXECUTABLE || (process.versions.electron ? process.execPath : require('electron')));
  const application = await _electron.launch({
    executablePath,
    cwd: path.dirname(executablePath),
    args: [desktop, '--disable-gpu', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
    env,
    timeout: 30000,
  });
  const report = {};
  try {
    const page = await application.firstWindow();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(() => document.querySelector('#modelDetail').textContent.includes('.bundle'));

    await application.evaluate(({ipcMain}, protocolPath) => {
      const requireForTest = process.getBuiltinModule('node:module').createRequire(protocolPath);
      const {validateInputAudio} = requireForTest(protocolPath);
      globalThis.audioIntegrationTest = {packets: [], connections: 0, interrupts: 0, errors: []};
      ipcMain.removeHandler('voice:connect');
      ipcMain.handle('voice:connect', event => {
        globalThis.audioIntegrationTest.connections += 1;
        // Match the initial cleanup event emitted by the actual connection path.
        event.sender.send('voice:event', {type: 'state', state: 'disconnected'});
        event.sender.send('voice:event', {type: 'state', state: 'loading'});
        return {state: 'loading', backend: 'trt_rtx'};
      });
      ipcMain.removeHandler('voice:interrupt');
      ipcMain.handle('voice:interrupt', () => {
        globalThis.audioIntegrationTest.interrupts += 1;
        // Deliberately delay native acknowledgment so late output can be tested.
        return {accepted: true};
      });
      ipcMain.on('voice:audio', (_event, input) => {
        try {
          const packet = validateInputAudio(input);
          let energy = 0;
          for (const value of packet.samples) energy += value * value;
          globalThis.audioIntegrationTest.packets.push({
            time: performance.now(), sampleRate: packet.sampleRate, count: packet.samples.length,
            rms: Math.sqrt(energy / packet.samples.length),
          });
        } catch (error) { globalThis.audioIntegrationTest.errors.push(error.message); }
      });
    }, path.join(desktop, 'protocol.js'));

    await page.evaluate(() => {
      const diagnostics = {tracks: [], contexts: [], sources: [], analysers: []};
      window.audioIntegrationTest = diagnostics;
      const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async constraints => {
        const stream = await getUserMedia(constraints);
        diagnostics.tracks.push(...stream.getTracks());
        return stream;
      };
      const NativeAudioContext = window.AudioContext;
      window.AudioContext = class extends NativeAudioContext {
        constructor(options) { super(options); diagnostics.contexts.push(this); }
        createAnalyser() {
          const analyser = super.createAnalyser();
          diagnostics.analysers.push(analyser);
          return analyser;
        }
        createBufferSource() {
          const source = super.createBufferSource();
          const record = {context: this, source, ended: false, stopped: false};
          diagnostics.sources.push(record);
          const start = source.start.bind(source);
          const stop = source.stop.bind(source);
          source.start = (when, ...args) => {
            record.when = when; record.at = this.currentTime;
            record.duration = source.buffer.duration; record.sampleRate = source.buffer.sampleRate;
            return start(when, ...args);
          };
          source.stop = (...args) => { record.stopped = true; return stop(...args); };
          source.addEventListener('ended', () => { record.ended = true; });
          return source;
        }
      };
    });

    const mainData = () => application.evaluate(() => globalThis.audioIntegrationTest);
    const emit = event => application.evaluate(({BrowserWindow}, packet) => {
      BrowserWindow.getAllWindows()[0].webContents.send('voice:event', packet);
    }, event);
    const emitNative = packet => application.evaluate(({BrowserWindow}, {packet, protocolPath}) => {
      const {normalizeEvent} = process.getBuiltinModule('node:module').createRequire(protocolPath)(protocolPath);
      for (const event of normalizeEvent(packet)) BrowserWindow.getAllWindows()[0].webContents.send('voice:event', event);
    }, {packet, protocolPath: path.join(desktop, 'protocol.js')});
    const audioPacket = (sampleRate = 48000) => ({
      type: 'audio', sampleRate,
      samples: Array.from({length: Math.round(sampleRate * .08)}, (_, i) => .1 * Math.sin(i * 2 * Math.PI * 440 / sampleRate)),
    });

    await page.click('#connectButton');
    await page.waitForFunction(() => window.audioIntegrationTest.contexts.length === 1);
    await page.waitForFunction(() => document.querySelector('#sessionState').textContent === 'Waking up Nemotron');
    await page.waitForTimeout(350);
    assert.equal((await mainData()).connections, 1, 'The real preload must reach the Electron connection handler.');
    assert.equal((await mainData()).packets.length, 0, 'Microphone data must wait until the native engines are ready.');
    await emit({type: 'state', state: 'listening', inputSampleRate: 16000});
    await page.waitForTimeout(3200);
    const captured = (await mainData()).packets;
    assert(captured.length >= 150, `Expected continuous 20 ms microphone capture, got ${captured.length} packets.`);
    assert(captured.every(packet => packet.sampleRate === 16000 && packet.count === 320), 'Capture must emit 20 ms mono packets at 16 kHz, before the native 80 ms silence deadline.');
    assert(captured.some(packet => packet.rms > .0001), 'The fake microphone signal must reach the native IPC endpoint.');
    const audioSeconds = (captured.length - 1) * 320 / 16000;
    const wallSeconds = (captured.at(-1).time - captured[0].time) / 1000;
    assert(Math.abs(audioSeconds - wallSeconds) < .2, `Capture clock drift: audio=${audioSeconds}s, wall=${wallSeconds}s.`);
    const intervalsMs = captured.slice(1).map((packet, index) => packet.time - captured[index].time).sort((a, b) => a - b);
    const p95IntervalMs = intervalsMs[Math.floor(intervalsMs.length * .95)];
    const p99IntervalMs = intervalsMs[Math.floor(intervalsMs.length * .99)];
    const maxIntervalMs = intervalsMs.at(-1);
    assert(p99IntervalMs < 60 && maxIntervalMs < 80, `Capture must arrive ahead of the native 80 ms silence deadline; p99=${p99IntervalMs} ms, max=${maxIntervalMs} ms.`);
    report.capture = {packets: captured.length, sampleRate: captured[0].sampleRate, samplesPerPacket: captured[0].count, audioSeconds, wallSeconds, p95IntervalMs, p99IntervalMs, maxIntervalMs};

    // Replay actual normalized model events: user utterances cross agent epochs
    // while their text remains one evolving snapshot. They must not split into
    // duplicate rows simply because the assistant starts or finishes speaking.
    const transcriptTrace = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'voicechat-epoch-transcripts.json'), 'utf8'));
    await application.evaluate(({BrowserWindow}, events) => {
      const contents = BrowserWindow.getAllWindows()[0].webContents;
      for (const event of events) contents.send('voice:event', event);
    }, transcriptTrace.events);
    await page.waitForFunction(() => document.querySelectorAll('.transcript-entry').length >= 6);
    const displayed = await page.locator('.transcript-entry').evaluateAll(entries => entries.map(entry => ({role: entry.classList.contains('assistant') ? 'assistant' : 'user', text: entry.querySelector('.entry-text').textContent, partial: entry.querySelector('.entry-text').classList.contains('partial')})));
    assert.deepEqual(displayed.filter(entry => entry.role === 'user').map(entry => entry.text), transcriptTrace.expectedUserTexts, 'Actual user partial/final snapshots must stay in the same row across agent epochs.');
    assert.deepEqual(displayed.filter(entry => entry.role === 'assistant').map(entry => entry.text), transcriptTrace.expectedAssistantTexts);
    assert(displayed.every(entry => !entry.partial), 'Completed actual transcripts must not retain a partial cursor.');
    report.actualTranscriptTrace = {events: transcriptTrace.events.length, displayedRows: displayed.length};

    await page.click('#muteButton');
    await page.waitForTimeout(180);
    const muteStart = (await mainData()).packets.length;
    await page.waitForTimeout(400);
    const muted = (await mainData()).packets.slice(muteStart);
    assert(muted.length >= 15 && muted.every(packet => packet.rms === 0), 'Muting must keep the 20 ms capture clock running with exact silence.');
    await page.click('#muteButton');

    // Nonuniform arrivals model GPU/IPC jitter. They should still form one
    // continuous audio timeline after a two-frame startup cushion.
    await emit(audioPacket());
    await page.waitForTimeout(100);
    await emit(audioPacket());
    await page.waitForTimeout(60);
    await emit(audioPacket(24000));
    await page.waitForTimeout(60);
    await emit(audioPacket());
    let sources = await page.evaluate(() => audioIntegrationTest.sources.map(({when, at, duration, sampleRate}) => ({when, at, duration, sampleRate})));
    assert.equal(sources.length, 4);
    assert(sources[0].when - sources[0].at >= .15, 'Playback needs a 160 ms initial jitter cushion.');
    assert(sources[0].when - sources[0].at <= .18, 'Playback must not add unbounded startup delay.');
    for (let i = 1; i < sources.length; i += 1) {
      assert(Math.abs(sources[i].when - sources[i - 1].when - sources[i - 1].duration) < .00001, 'Jittery input packets must play contiguously, including a sample-rate change.');
    }
    const audibleEnergy = await page.evaluate(() => {
      const analyser = audioIntegrationTest.analysers.at(-1);
      const samples = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(samples);
      return samples.reduce((total, value) => total + value * value, 0) / samples.length;
    });
    assert(audibleEnergy > .0001, 'Scheduled PCM must actually reach the output analyser.');
    report.playback = {initialPrebufferMs: (sources[0].when - sources[0].at) * 1000, continuousPackets: sources.length, outputEnergy: audibleEnergy};

    await emitNative({type: 'event', kind: 'user_speech_started', epoch: 8, sequence: 0});
    assert(await page.evaluate(() => audioIntegrationTest.sources.every(record => !record.stopped)), 'Speech detection alone must not cut playback; native yield decides automatic barge-in.');
    await emitNative({type: 'event', kind: 'yielded', epoch: 9, sequence: 0});
    await page.waitForTimeout(60);
    assert(await page.evaluate(() => audioIntegrationTest.sources.every(record => record.ended || record.stopped)), 'Barge-in must stop both playing and future scheduled sources.');
    const flushedEnergy = await page.evaluate(() => {
      const analyser = audioIntegrationTest.analysers.at(-1);
      const samples = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(samples);
      return samples.reduce((total, value) => total + value * value, 0) / samples.length;
    });
    assert.equal(flushedEnergy, 0, 'Output must be silent after the flush has drained through the audio graph.');

    // Recovery after flush and after a natural underrun both rebuild the cushion.
    await emit(audioPacket());
    await page.waitForTimeout(350);
    await emit(audioPacket());
    sources = await page.evaluate(() => audioIntegrationTest.sources.map(({when, at}) => ({when, at})));
    for (const source of sources.slice(-2)) assert(source.when - source.at >= .15 && source.when - source.at <= .18, 'Flush/underrun recovery must rebuild the two-frame buffer.');

    // The explicit control silences local PCM before native acknowledgment and
    // rejects audio/text already in flight, without losing an ongoing user turn.
    await emit(audioPacket());
    await emit({type: 'transcript', role: 'user', text: 'Please stop', final: false, epoch: 10});
    const sourcesBeforeInterrupt = await page.evaluate(() => audioIntegrationTest.sources.length);
    const inputBeforeInterrupt = (await mainData()).packets.length;
    await page.click('#interruptButton');
    assert(await page.evaluate(() => audioIntegrationTest.sources.every(record => record.ended || record.stopped)), 'Stop speaking must flush audible and future PCM immediately, before native acknowledgment.');
    assert.equal((await mainData()).interrupts, 1, 'The button must reach the isolated preload IPC control.');
    await emitNative({type: 'event', kind: 'yielded', epoch: 11, sequence: 0});
    await emit(audioPacket());
    await emit({type: 'transcript', role: 'assistant', text: 'STALE INTERRUPTED OUTPUT', final: false, delta: true, epoch: 10});
    await emit({type: 'transcript', role: 'user', text: 'Please stop this response.', final: true, epoch: 10});
    await page.keyboard.press('i');
    await page.waitForTimeout(220);
    assert.equal((await mainData()).interrupts, 1, 'Repeated input while the interrupt is pending must not enqueue duplicate controls.');
    assert.equal(await page.evaluate(() => audioIntegrationTest.sources.length), sourcesBeforeInterrupt, 'Late PCM must not restart the interrupted response.');
    assert(!(await page.locator('#transcriptScroll').textContent()).includes('STALE INTERRUPTED OUTPUT'), 'Late assistant text must not reopen the interrupted turn.');
    const userAfterInterrupt = await page.locator('.transcript-entry.user .entry-text').allTextContents();
    assert.equal(userAfterInterrupt.filter(text => text.startsWith('Please stop')).length, 1, 'Interrupt must preserve the evolving user transcript.');
    assert(userAfterInterrupt.includes('Please stop this response.'));
    assert((await mainData()).packets.length >= inputBeforeInterrupt + 8, 'Microphone IPC must continue while interruption is pending.');
    assert(await page.evaluate(() => audioIntegrationTest.tracks.every(track => track.readyState === 'live') && audioIntegrationTest.contexts.every(context => context.state === 'running')), 'Interrupt must preserve the microphone and audio context.');
    // The bridge acknowledges its cached-state reset, then delivers the native
    // reset event. RNNT partials from that abandoned state must be finalized.
    await emit({type: 'transcript', role: 'user', text: 'Old context request still arriving', final: false, epoch: 11});
    assert.equal(await page.locator('.transcript-entry.user .entry-text.partial').textContent(), 'Old context request still arriving');
    const inputBeforeReset = (await mainData()).packets.length;
    await emit({type: 'flush', reason: 'interrupt', interruptStatus: 'context_reset'});
    await emitNative({type: 'event', kind: 'reset', epoch: 12, sequence: 0});
    await page.waitForFunction(() => [...document.querySelectorAll('.transcript-notice')].some(notice => notice.textContent === 'Conversation refreshed. Earlier history was cleared.'));
    assert.equal(await page.locator('.transcript-entry.user .entry-text.partial').count(), 0, 'The native reset must finalize the old RNNT partial.');
    assert((await page.locator('.transcript-entry.user .entry-text').allTextContents()).includes('Old context request still arriving'), 'Reset must keep the abandoned transcript visible as completed history.');
    await emit({type: 'transcript', role: 'user', text: 'A fresh question after reset', final: false, epoch: 12});
    assert.equal(await page.locator('.transcript-entry.user .entry-text.partial').textContent(), 'A fresh question after reset', 'A fresh RNNT snapshot must appear in its own visible row after reset.');
    await emit({type: 'transcript', role: 'user', text: 'A fresh question after reset.', final: true, epoch: 12});
    await page.waitForTimeout(220);
    assert((await mainData()).packets.length >= inputBeforeReset + 8, 'The reset barrier must preserve continuous microphone IPC.');
    assert(await page.evaluate(() => audioIntegrationTest.tracks.every(track => track.readyState === 'live') && audioIntegrationTest.contexts.length === 1 && audioIntegrationTest.contexts.every(context => context.state === 'running')), 'The reset barrier must retain the existing microphone and audio context.');
    await emit(audioPacket());
    assert.equal(await page.evaluate(() => audioIntegrationTest.sources.length), sourcesBeforeInterrupt + 1, 'A new response must play after the native interruption barrier.');

    await page.click('#streamButton');
    await page.keyboard.press('i');
    assert.equal((await mainData()).interrupts, 2, 'The interruption shortcut must work while stream mode hides the controls.');
    assert(await page.evaluate(() => audioIntegrationTest.sources.every(record => record.ended || record.stopped)));
    await emit({type: 'flush', reason: 'interrupt', interruptStatus: 'context_reset'});
    await emitNative({type: 'event', kind: 'reset', epoch: 13, sequence: 0});
    await page.keyboard.press('Escape');
    report.interruption = {nativeControls: (await mainData()).interrupts, inFlightOutputSuppressed: true, microphoneKeptActive: true, contextResetBarrier: true, abandonedPartialFinalized: true, newPartialVisible: true, refreshNoticeVisible: true, streamShortcut: true};

    await page.click('#connectButton');
    await page.waitForFunction(() => audioIntegrationTest.tracks.every(track => track.readyState === 'ended') && audioIntegrationTest.contexts.every(context => context.state === 'closed'));
    const stoppedCount = (await mainData()).packets.length;
    await page.waitForTimeout(300);
    assert.equal((await mainData()).packets.length, stoppedCount, 'Disconnect must stop microphone IPC completely.');
    assert(await page.evaluate(() => audioIntegrationTest.sources.every(record => record.ended || record.stopped)), 'Disconnect must also release queued output sources.');

    // Reconnecting must create a fresh capture clock without reviving old tracks.
    await page.click('#connectButton');
    await page.waitForFunction(() => audioIntegrationTest.contexts.length === 2);
    await page.waitForTimeout(200);
    await emit({type: 'state', state: 'listening', inputSampleRate: 16000});
    await page.waitForTimeout(450);
    assert((await mainData()).packets.length >= stoppedCount + 18, 'A fresh session must resume 20 ms microphone capture.');
    await emit({type: 'error', fatal: true, message: 'Injected native failure for microphone cleanup verification.'});
    await page.waitForFunction(() => audioIntegrationTest.tracks.every(track => track.readyState === 'ended') && audioIntegrationTest.contexts.every(context => context.state === 'closed'));
    assert.equal(await page.locator('#sessionHint').textContent(), 'Injected native failure for microphone cleanup verification.');
    const finalMain = await mainData();
    assert.deepEqual(finalMain.errors, [], 'All packets must satisfy the actual native input validator.');
    assert.deepEqual(errors, [], 'No renderer exceptions are allowed.');
    report.checks = ['loading gate', 'real 16 kHz IPC', '20 ms capture timing', 'actual transcript trace across epochs', 'mute silence', '160 ms prebuffer', 'jitter continuity', 'mixed output sample rates', 'audible PCM', 'native yield flush', 'underrun recovery', 'explicit interruption before acknowledgment', 'late interrupted output suppression', 'user transcript and microphone retained', 'context reset finalizes abandoned partial', 'fresh partial visible after reset', 'context refresh notice', 'stream mode interrupt shortcut', 'microphone release', 'reconnect', 'fatal failure cleanup'];
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await application.close();
    // No recursive cleanup: the isolated temporary profile is retained for logs.
    console.log(`Isolated test profile: ${workspace}`);
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
