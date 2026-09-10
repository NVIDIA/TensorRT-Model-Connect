'use strict';

// Opt-in, expensive packaged-app test. Requires the installed native runtime,
// actual TensorRT-RTX bundle, and a recorded 16 kHz test question. No inference,
// IPC handler, or application API is mocked. Only microphone input is a fixture.
const {_electron} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const workspace = path.resolve(process.env.VOICE_LAB_WORKSPACE || path.resolve(__dirname, '../../../../..'));
const fixture = path.resolve(process.env.VOICE_LAB_AUDIO_FIXTURE || path.join(workspace, 'logs', 'voice-clean-question.wav'));
const receiptPath = path.join(workspace, 'logs', 'electron-real-model-verification.json');
const screenshotPath = path.join(workspace, 'logs', 'voice-lab-real-conversation.png');
const executable = path.resolve(process.env.VOICE_LAB_APP_EXECUTABLE || path.join(workspace, 'Nemotron Voice Lab', 'Nemotron Voice Lab.exe'));
const fixtureRepeats = Number(process.env.VOICE_LAB_FIXTURE_REPEATS || 1);
const fixtureInterval = 28;
assert(Number.isInteger(fixtureRepeats) && fixtureRepeats >= 1 && fixtureRepeats <= 5);
const env = {...process.env, VOICE_LAB_WORKSPACE: workspace};
delete env.ELECTRON_RUN_AS_NODE;

function writeWave(file, samples, sampleRate) {
  const bytes = Buffer.alloc(44 + samples.length * 4);
  bytes.write('RIFF', 0); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write('WAVEfmt ', 8);
  bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(3, 20); bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(sampleRate, 24); bytes.writeUInt32LE(sampleRate * 4, 28);
  bytes.writeUInt16LE(4, 32); bytes.writeUInt16LE(32, 34); bytes.write('data', 36);
  bytes.writeUInt32LE(samples.length * 4, 40);
  samples.forEach((value, index) => bytes.writeFloatLE(value, 44 + index * 4));
  fs.writeFileSync(file, bytes);
}

async function run() {
  assert(fs.statSync(fixture, {throwIfNoEntry: false})?.isFile(), 'Set VOICE_LAB_AUDIO_FIXTURE to an existing WAV recording of a test question.');
  fs.mkdirSync(path.dirname(receiptPath), {recursive: true});
  const receipt = {passed: false, executable, inputFixture: fixture, microphone: 'Recorded WAV routed through a test-only MediaStreamDestination; no user microphone captured.', inference: 'Unmodified packaged Electron app, production preload/IPC/native bridge, actual GPU TensorRT-RTX model.', startedAt: new Date().toISOString()};
  const application = await _electron.launch({executablePath: executable, cwd: path.dirname(executable), args: [], env, timeout: 45000});
  let page;
  try {
    page = await application.firstWindow();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(() => document.querySelector('#modelDetail').textContent.includes('.bundle'));
    receipt.config = await page.evaluate(() => window.voiceLab.getConfig());
    assert(fs.statSync(receipt.config.bundlePath).size > 1024 * 1024 * 1024, 'The real model bundle must exist.');
    assert(fs.existsSync(receipt.config.bridgePath), 'The real native bridge must exist.');

    // Observe the actual production IPC listener without replacing any handler.
    await application.evaluate(({ipcMain}) => {
      globalThis.realModelInputAudit = [];
      ipcMain.on('voice:audio', (_event, packet) => {
        let energy = 0;
        for (const value of packet.samples) energy += value * value;
        globalThis.realModelInputAudit.push({at: Date.now(), sampleRate: packet.sampleRate, count: packet.samples.length, rms: Math.sqrt(energy / packet.samples.length)});
      });
    });

    await page.evaluate(async base64 => {
      const test = {events: [], sources: [], outputSamples: [], fixture: {}, tracks: []};
      window.realModelTest = test;
      const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
      window.voiceLab.onEvent(event => {
        const record = {...event, at: Date.now()};
        if (event.type === 'audio') {
          record.sampleCount = event.samples.length;
          record.rms = Math.sqrt(event.samples.reduce((sum, value) => sum + value * value, 0) / event.samples.length);
          delete record.samples;
          test.outputSamples.push(...event.samples);
        }
        if (event.type !== 'metrics') test.events.push(record);
      });
      const originalStart = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(when = 0, ...args) {
        if (this.context !== test.fixture.context) {
          const record = {at: Date.now(), when, contextTime: this.context.currentTime, duration: this.buffer.duration, stopped: false, ended: false};
          test.sources.push(record);
          const stop = this.stop.bind(this);
          this.stop = (...stopArgs) => { record.stopped = true; return stop(...stopArgs); };
          this.addEventListener('ended', () => { record.ended = true; });
        }
        return originalStart.call(this, when, ...args);
      };
      navigator.mediaDevices.getUserMedia = async () => {
        const context = new AudioContext({sampleRate: 48000, latencyHint: 'interactive'});
        const destination = context.createMediaStreamDestination();
        const silence = context.createConstantSource();
        silence.offset.value = 0;
        silence.connect(destination); silence.start();
        test.fixture = {context, destination, buffer: await context.decodeAudioData(bytes.buffer.slice(0))};
        test.tracks.push(...destination.stream.getTracks());
        await context.resume();
        return destination.stream;
      };
    }, fs.readFileSync(fixture).toString('base64'));

    receipt.connectRequestedAt = Date.now();
    await page.click('#connectButton');
    console.log('Packaged app requested the real native TensorRT-RTX session. Waiting for readiness.');
    await page.waitForFunction(() => realModelTest.events.some(event => event.type === 'state' && event.state === 'listening') || realModelTest.events.some(event => event.type === 'error'), null, {timeout: 180000});
    const initial = await page.evaluate(() => realModelTest.events);
    assert(!initial.some(event => event.type === 'error'), JSON.stringify(initial.filter(event => event.type === 'error')));
    receipt.readyAt = initial.find(event => event.type === 'state' && event.state === 'listening').at;
    receipt.loadSeconds = (receipt.readyAt - receipt.connectRequestedAt) / 1000;
    receipt.fixtureTiming = await page.evaluate(({repeats, interval}) => {
      const fixture = realModelTest.fixture;
      const delay = 8;
      const record = {scheduledAt: Date.now(), delaySeconds: delay, durationSeconds: fixture.buffer.duration, scheduledStartAt: Date.now() + delay * 1000, repeats, intervalSeconds: interval};
      fixture.timing = record;
      for (let index = 0; index < repeats; index++) {
        const source = fixture.context.createBufferSource();
        source.buffer = fixture.buffer; source.connect(fixture.destination);
        source.onended = () => { record.endedAt = Date.now(); };
        source.start(fixture.context.currentTime + delay + index * interval);
      }
      return record;
    }, {repeats: fixtureRepeats, interval: fixtureInterval});
    console.log(`Native ready in ${receipt.loadSeconds.toFixed(2)}s. Recorded fixture starts in 8s; ${fixtureRepeats} repetition(s), ${fixtureInterval}s apart.`);

    const earliestFinish = receipt.fixtureTiming.scheduledStartAt + ((fixtureRepeats - 1) * fixtureInterval + receipt.fixtureTiming.durationSeconds + 15) * 1000;
    const deadline = receipt.readyAt + 150000 + (fixtureRepeats - 1) * fixtureInterval * 1000;
    let finalState;
    let lastUpdate = 0;
    while (Date.now() < deadline) {
      await page.waitForTimeout(1000);
      finalState = await page.evaluate(() => {
        const events = realModelTest.events;
        const audio = events.filter(event => event.type === 'audio');
        return {errors: events.filter(event => event.type === 'error'), transcripts: events.filter(event => event.type === 'transcript'), audioPackets: audio.length, lastAudioAt: audio.at(-1)?.at || 0, allPlaybackFinished: realModelTest.sources.every(source => source.ended || source.stopped), backendStatus: document.querySelector('#backendStatus').textContent, sessionHint: document.querySelector('#sessionHint').textContent};
      });
      if (finalState.errors.length) throw new Error(JSON.stringify(finalState.errors));
      if (finalState.backendStatus !== 'Connected locally') throw new Error(finalState.sessionHint);
      if (Date.now() - lastUpdate > 12000) {
        lastUpdate = Date.now();
        console.log(JSON.stringify({secondsSinceReady: Math.round((Date.now() - receipt.readyAt) / 1000), audioPackets: finalState.audioPackets, finalTranscripts: finalState.transcripts.filter(event => event.final).map(event => ({role: event.role, text: event.text}))}));
      }
      const answered = finalState.transcripts.some(event => event.role === 'assistant' && event.final && /paris/i.test(event.text));
      if (answered && Date.now() >= earliestFinish && Date.now() - finalState.lastAudioAt > 8000 && finalState.allPlaybackFinished) break;
    }
    receipt.events = await page.evaluate(() => realModelTest.events);
    receipt.outputScheduling = await page.evaluate(() => realModelTest.sources);
    receipt.maxScheduledAudioSeconds = Math.max(0, ...receipt.outputScheduling.map(source => source.when + source.duration - source.contextTime));
    receipt.inputPackets = await application.evaluate(() => globalThis.realModelInputAudit);
    const inputGaps = receipt.inputPackets.slice(1).map((packet, index) => packet.at - receipt.inputPackets[index].at).sort((a, b) => a - b);
    receipt.inputCadence = {packetSamples: 320, packetDurationMs: 20, medianGapMs: inputGaps[Math.floor(inputGaps.length / 2)], p99GapMs: inputGaps[Math.floor(inputGaps.length * .99)], maxGapMs: inputGaps.at(-1)};
    receipt.renderedTranscript = await page.locator('.transcript-entry').evaluateAll(entries => entries.map(entry => ({role: entry.classList.contains('assistant') ? 'assistant' : 'user', text: entry.querySelector('.entry-text').textContent, partial: entry.querySelector('.entry-text').classList.contains('partial')})));
    receipt.fixtureTiming = await page.evaluate(() => realModelTest.fixture.timing);
    receipt.errors = pageErrors.concat(receipt.events.filter(event => event.type === 'error').map(event => event.message));
    const output = await page.evaluate(() => realModelTest.outputSamples);
    receipt.outputAudio = path.join(workspace, 'logs', 'electron-real-model-output.wav');
    writeWave(receipt.outputAudio, output, 48000);
    receipt.outputSamples = output.length;
    const speechInput = receipt.inputPackets.filter(packet => packet.rms > .0001);
    const audioEvents = receipt.events.filter(event => event.type === 'audio');
    const firstAudio = audioEvents[0];
    const firstNonSilentAudio = audioEvents.find(event => event.rms > .0001);
    const lastAudio = audioEvents.at(-1);
    receipt.timing = {firstInputSignalAt: speechInput[0]?.at, lastInputSignalAt: speechInput.at(-1)?.at, firstNativeAudioAt: firstAudio?.at, firstNonSilentNativeAudioAt: firstNonSilentAudio?.at, nonSilentRmsThreshold: .0001, lastNativeAudioAt: lastAudio?.at, firstAudioAfterInputStartMs: firstAudio && speechInput.length ? firstAudio.at - speechInput[0].at : null, firstNonSilentAudioAfterInputStartMs: firstNonSilentAudio && speechInput.length ? firstNonSilentAudio.at - speechInput[0].at : null, lastAudioAfterInputEndMs: lastAudio && speechInput.length ? lastAudio.at - speechInput.at(-1).at : null};
    receipt.playbackGaps = receipt.outputScheduling.flatMap((source, index, sources) => {
      if (!index) return [];
      const gapMs = (source.when - sources[index - 1].when - sources[index - 1].duration) * 1000;
      if (gapMs <= 20) return [];
      const fromEpoch = audioEvents[index - 1]?.epoch;
      const toEpoch = audioEvents[index]?.epoch;
      const category = sources[index - 1].stopped ? 'interruption' : fromEpoch !== toEpoch ? 'turn_boundary' : 'same_turn_underrun';
      return [{sourceIndex: index, gapMs, fromEpoch, toEpoch, category}];
    });
    receipt.playbackRebuffers = receipt.playbackGaps.length;
    receipt.sameTurnUnderruns = receipt.playbackGaps.filter(gap => gap.category === 'same_turn_underrun').length;
    await page.click('#streamButton');
    await page.waitForTimeout(500);
    await application.evaluate(({BrowserWindow}) => {
      const window = BrowserWindow.getAllWindows()[0];
      window.setFullScreen(false);
      window.setContentSize(1920, 1080);
    });
    await page.waitForTimeout(6500);
    await page.screenshot({path: screenshotPath});
    receipt.screenshot = screenshotPath;
    receipt.screenshotDescription = 'Actual packaged app and real local TensorRT-RTX inference; microphone input was the recorded test fixture.';
    assert.equal(await page.locator('#rehearsalBadge').isVisible(), false);
    assert.match(await page.locator('#backendStatus').textContent(), /Connected locally/);
    assert.equal(receipt.errors.length, 0, JSON.stringify(receipt.errors));
    assert(receipt.inputPackets.length > 100 && receipt.inputPackets.every(packet => packet.sampleRate === 16000 && packet.count === 320));
    assert(receipt.maxScheduledAudioSeconds < 2, `Short replies must not accumulate a growing playback queue: ${receipt.maxScheduledAudioSeconds.toFixed(3)}s queued.`);
    assert(output.length > 48000 && receipt.events.some(event => event.type === 'audio' && event.rms > .001));
    assert(lastAudio.at >= receipt.fixtureTiming.scheduledStartAt + (fixtureRepeats - 1) * fixtureInterval * 1000, 'The model must still produce audio for the final repetition.');
    assert(receipt.events.some(event => event.type === 'transcript' && event.role === 'assistant' && event.final && /paris/i.test(event.text)), 'The real model must finish an answer containing Paris.');
    const finalUserTexts = receipt.events.filter(event => event.type === 'transcript' && event.role === 'user' && event.final).map(event => event.text);
    assert.deepEqual(receipt.renderedTranscript.filter(entry => entry.role === 'user').map(entry => entry.text), finalUserTexts, 'User partials spanning agent epochs must resolve to one displayed row per final utterance.');
    receipt.passed = true;
  } catch (error) {
    receipt.failure = error.message;
    if (page && !page.isClosed()) {
      receipt.currentState = await page.evaluate(() => ({state: document.querySelector('#sessionState')?.textContent, hint: document.querySelector('#sessionHint')?.textContent, events: window.realModelTest?.events})).catch(() => null);
      await page.screenshot({path: path.join(workspace, 'logs', 'voice-lab-real-failure.png')}).catch(() => {});
    }
    throw error;
  } finally {
    if (page && !page.isClosed()) {
      await page.evaluate(async () => { await window.voiceLab.disconnect(); if (window.realModelTest?.fixture?.context?.state !== 'closed') await window.realModelTest?.fixture?.context?.close(); }).catch(() => {});
    }
    await application.close();
    receipt.finishedAt = new Date().toISOString();
    fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
    console.log(JSON.stringify({passed: receipt.passed, loadSeconds: receipt.loadSeconds, timing: receipt.timing, outputSamples: receipt.outputSamples, maxScheduledAudioSeconds: receipt.maxScheduledAudioSeconds, playbackGaps: receipt.playbackGaps, sameTurnUnderruns: receipt.sameTurnUnderruns, failure: receipt.failure, receipt: receiptPath, screenshot: receipt.screenshot}));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
