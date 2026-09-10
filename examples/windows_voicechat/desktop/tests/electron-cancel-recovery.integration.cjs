'use strict';

// Focused real-model regression for the failure after Stop speaking in an aged
// context. This supplements the strict nine-minute conversation soak; it does
// not replace that test or alter its acceptance criteria.
const {_electron} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const workspace = path.resolve(process.env.VOICE_LAB_WORKSPACE || path.resolve(__dirname, '../../../../..'));
const manifestPath = path.resolve(process.env.VOICE_LAB_SOAK_MANIFEST || path.join(workspace, 'logs', 'voice-soak-fixtures', 'manifest.json'));
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8').replace(/^\uFEFF/, ''));
const turns = [manifest.turns[8], manifest.turns[9], manifest.turns[10]];
const receiptPath = path.resolve(process.env.VOICE_LAB_CANCEL_RECEIPT || path.join(workspace, 'logs', 'electron-cancel-recovery.json'));
const executable = path.resolve(process.env.VOICE_LAB_APP_EXECUTABLE || path.join(workspace, 'Nemotron Voice Lab', 'Nemotron Voice Lab.exe'));
const env = {...process.env, VOICE_LAB_WORKSPACE: workspace};
delete env.ELECTRON_RUN_AS_NODE;

async function run() {
  fs.mkdirSync(path.dirname(receiptPath), {recursive: true});
  const receipt = {passed: false, startedAt: new Date().toISOString(), executable, warmupSeconds: 45, microphone: 'Three Windows SAPI fixtures through a test-only MediaStreamDestination; no user microphone.', inference: 'Production packaged renderer, worklet, preload, IPC, bridge and real TensorRT-RTX model.', questions: []};
  const fixtures = turns.map(turn => {
    const bytes = fs.readFileSync(path.resolve(path.dirname(manifestPath), turn.audio));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), turn.sha256);
    return bytes.toString('base64');
  });
  const application = await _electron.launch({executablePath: executable, cwd: path.dirname(executable), env, timeout: 45000});
  let page;
  let readyAt = 0;
  let lastProgress = 0;
  async function state() {
    const value = await page.evaluate(() => {
      const test = cancelRecoveryTest;
      return {events: test.events.filter(event => event.type !== 'audio'), replies: Object.values(test.replies), active: test.sources.filter(source => !source.ended && !source.stopped && source.context.currentTime >= source.when && source.context.currentTime < source.when + source.duration && source.rms > .0001).map(({context, ...source}) => source), tracks: test.fixture.destination.stream.getTracks().map(track => track.readyState), maxPlaybackLeadSeconds: test.maxPlaybackLeadSeconds};
    });
    assert(!value.events.some(event => event.type === 'error'), JSON.stringify(value.events.filter(event => event.type === 'error')));
    assert(value.maxPlaybackLeadSeconds < 2, `Playback lead exceeded 2 seconds: ${value.maxPlaybackLeadSeconds}`);
    if (readyAt && Date.now() - lastProgress >= 10000) {
      lastProgress = Date.now();
      console.log(JSON.stringify({seconds: Math.round((Date.now() - readyAt) / 1000), questions: receipt.questions.length, lastReply: value.replies.at(-1)?.text, rollovers: value.events.filter(event => event.type === 'context_rolled').length}));
    }
    return value;
  }
  async function until(predicate, timeoutMs, description) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const snapshot = await state();
      const result = predicate(snapshot);
      if (result) return result;
      await page.waitForTimeout(40);
    }
    throw new Error(`Timed out waiting for ${description}.`);
  }
  async function pauseTo(targetAt) {
    while (Date.now() < targetAt) { await state(); await page.waitForTimeout(Math.min(1000, targetAt - Date.now())); }
  }
  async function speak(index) {
    return page.evaluate(index => {
      const test = cancelRecoveryTest;
      const source = test.fixture.context.createBufferSource();
      source.buffer = test.fixture.buffers[index]; source.connect(test.fixture.destination);
      const timing = {index, startAt: Date.now() + 100, durationSeconds: source.buffer.duration};
      timing.endAt = timing.startAt + timing.durationSeconds * 1000;
      test.fixture.starts.push(timing); source.start(test.fixture.context.currentTime + .1);
      return timing;
    }, index);
  }
  async function expectAnswer(index, timing) {
    const expected = turns[index].expectedAny;
    const reply = await until(snapshot => snapshot.replies.find(reply => reply.startedAt >= timing.startAt && reply.final && expected.some(word => new RegExp(`\\b${word}\\b`, 'i').test(reply.text))), 45000, `${turns[index].id}: ${expected.join(' or ')}`);
    const snapshot = await state();
    assert(snapshot.events.some(event => event.type === 'transcript' && event.role === 'user' && event.final && event.at >= timing.startAt), 'Follow-up must have a recognized final user transcript.');
    receipt.questions.push({id: turns[index].id, prompt: turns[index].text, expectedAny: expected, timing, reply});
  }
  try {
    page = await application.firstWindow();
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(() => document.querySelector('#modelDetail').textContent.includes('.bundle'));
    await application.evaluate(({ipcMain}) => {
      globalThis.cancelRecoveryInput = [];
      ipcMain.on('voice:audio', (_event, packet) => globalThis.cancelRecoveryInput.push({at: Date.now(), count: packet.samples.length, sampleRate: packet.sampleRate}));
    });
    await page.evaluate(async fixtures => {
      const test = {events: [], replies: {}, sources: [], unmatched: [], maxPlaybackLeadSeconds: 0, fixture: {}};
      window.cancelRecoveryTest = test;
      const start = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(when = 0, ...args) {
        if (this.context !== test.fixture.context) {
          const samples = this.buffer.getChannelData(0);
          const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
          const record = {id: test.sources.length, at: Date.now(), when, duration: this.buffer.duration, contextTime: this.context.currentTime, context: this.context, rms, stopped: false, ended: false};
          test.sources.push(record); test.unmatched.push(record);
          test.maxPlaybackLeadSeconds = Math.max(test.maxPlaybackLeadSeconds, when + record.duration - record.contextTime);
          const stop = this.stop.bind(this);
          this.stop = (...args) => { record.stopped = true; record.stoppedAt = Date.now(); return stop(...args); };
          this.addEventListener('ended', () => { record.ended = true; record.endedAt = Date.now(); });
        }
        return start.call(this, when, ...args);
      };
      voiceLab.onEvent(event => {
        const record = {...event, at: Date.now()};
        if (event.type === 'audio') {
          record.sampleCount = event.samples.length; delete record.samples;
          const source = test.unmatched.shift(); if (source) source.epoch = event.epoch;
        }
        if (event.type === 'transcript' && event.role === 'assistant') {
          const reply = test.replies[event.epoch] ||= {epoch: event.epoch, text: '', startedAt: record.at};
          reply.text = event.delta ? reply.text + event.text : event.text;
          reply.final = event.final; reply.updatedAt = record.at;
        }
        if (event.type !== 'metrics') test.events.push(record);
      });
      navigator.mediaDevices.getUserMedia = async () => {
        const context = new AudioContext({sampleRate: 48000, latencyHint: 'interactive'});
        const destination = context.createMediaStreamDestination();
        const silence = context.createConstantSource(); silence.offset.value = 0; silence.connect(destination); silence.start();
        test.fixture = {context, destination, buffers: [], starts: []};
        for (const fixture of fixtures) {
          const bytes = Uint8Array.from(atob(fixture), character => character.charCodeAt(0));
          test.fixture.buffers.push(await context.decodeAudioData(bytes.buffer.slice(0)));
        }
        await context.resume(); return destination.stream;
      };
    }, fixtures);
    await page.click('#connectButton');
    await page.waitForFunction(() => cancelRecoveryTest.events.some(event => event.backend === 'trt_rtx') || cancelRecoveryTest.events.some(event => event.type === 'error'), null, {timeout: 180000});
    readyAt = (await state()).events.find(event => event.backend === 'trt_rtx').at;
    receipt.readyAt = readyAt;
    await pauseTo(readyAt + 45000);
    const piano = await speak(0);
    await pauseTo(piano.endAt);
    const audible = await until(snapshot => snapshot.active.find(source => source.at >= piano.startAt), 15000, 'audible piano reply');
    receipt.interrupt = {requestedAt: Date.now(), oldEpoch: audible.epoch, audible};
    await page.click('#interruptButton');
    receipt.interrupt.ack = await until(snapshot => snapshot.events.find(event => event.type === 'flush' && event.reason === 'interrupt' && event.at >= receipt.interrupt.requestedAt), 10000, 'button interrupt acknowledgement');
    await pauseTo(receipt.interrupt.requestedAt + 3000);
    const egypt = await speak(1);
    await expectAnswer(1, egypt);
    receipt.interrupt.nativeReset = (await state()).events.find(event => event.type === 'flush' && event.reason === 'reset' && event.at >= receipt.interrupt.requestedAt);
    assert(receipt.interrupt.nativeReset, 'Stop must complete its native reset barrier and release the worker.');
    assert.equal(receipt.interrupt.ack.interruptStatus, 'context_reset');
    await pauseTo(Math.max(Date.now(), egypt.endAt) + 2000);
    const week = await speak(2);
    await expectAnswer(2, week);
    await pauseTo(readyAt + 90000);
    assert((await state()).tracks.every(track => track === 'live'));
    const packets = await application.evaluate(() => globalThis.cancelRecoveryInput);
    assert(packets.every(packet => packet.count === 320 && packet.sampleRate === 16000));
    assert(packets.filter(packet => packet.at > receipt.interrupt.requestedAt).length > 500, 'Microphone capture must continue for more than ten seconds after Stop.');
    receipt.passed = true;
  } catch (error) { receipt.failure = error.message; throw error; }
  finally {
    if (page && !page.isClosed()) {
      Object.assign(receipt, await page.evaluate(() => ({events: window.cancelRecoveryTest?.events, outputScheduling: window.cancelRecoveryTest?.sources.map(({context, ...source}) => source), fixtureTimings: window.cancelRecoveryTest?.fixture?.starts})).catch(() => ({})));
      receipt.inputPackets = await application.evaluate(() => globalThis.cancelRecoveryInput).catch(() => []);
      await page.screenshot({path: receiptPath.replace(/\.json$/, '.png')}).catch(() => {});
      await page.evaluate(async () => { await voiceLab.disconnect(); if (window.cancelRecoveryTest?.fixture?.context?.state !== 'closed') await cancelRecoveryTest.fixture.context.close(); }).catch(() => {});
    }
    await application.close();
    receipt.finishedAt = new Date().toISOString();
    fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
    console.log(JSON.stringify({passed: receipt.passed, failure: receipt.failure, receipt: receiptPath, replies: receipt.questions.map(question => question.reply.text)}));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
