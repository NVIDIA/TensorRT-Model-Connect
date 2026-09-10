'use strict';

// Opt-in real-model regression: only microphone input is a synthesized fixture.
// The packaged renderer, capture worklet, preload, IPC, native bridge, and RTX
// model all run normally. Do not run alongside an interactive GPU session.
const {_electron} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const workspace = path.resolve(process.env.VOICE_LAB_WORKSPACE || path.resolve(__dirname, '../../../../..'));
const manifestPath = path.resolve(process.env.VOICE_LAB_SOAK_MANIFEST || path.join(workspace, 'logs', 'voice-soak-fixtures', 'manifest.json'));
const receiptPath = path.resolve(process.env.VOICE_LAB_SOAK_RECEIPT || path.join(workspace, 'logs', 'electron-conversation-soak.json'));
const executable = path.resolve(process.env.VOICE_LAB_APP_EXECUTABLE || path.join(workspace, 'Nemotron Voice Lab', 'Nemotron Voice Lab.exe'));
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8').replace(/^\uFEFF/, ''));
const env = {...process.env, VOICE_LAB_WORKSPACE: workspace};
delete env.ELECTRON_RUN_AS_NODE;

function normalized(text) { return text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim(); }
function containsExpected(text, words) {
  const padded = ` ${normalized(text)} `;
  return words.some(word => padded.includes(` ${normalized(word)} `));
}
function assertNoOldStory(text, turn) {
  // Acknowledging "I will stop the story" is appropriate; resuming its
  // narrative is the failure. Long story-specific phrases avoid rejecting
  // a short acknowledgement or generic conversational phrasing.
  for (const phrase of turn.forbiddenContinuation || []) {
    assert(!normalized(text).includes(normalized(phrase)), `${turn.id} continued the abandoned story: ${text}`);
  }
}

async function run() {
  fs.mkdirSync(path.dirname(receiptPath), {recursive: true});
  assert.equal(manifest.format, 'voice-lab-multitopic-soak-v1');
  assert.equal(manifest.turns.length, 12);
  assert(new Set(manifest.turns.map(turn => normalized(turn.text))).size === 12, 'All spoken requests must differ.');
  const fixtures = manifest.turns.map(turn => {
    const bytes = fs.readFileSync(path.resolve(path.dirname(manifestPath), turn.audio));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), turn.sha256, `Fixture changed: ${turn.id}`);
    return bytes.toString('base64');
  });
  const receipt = {
    passed: false, executable, manifestPath, startedAt: new Date().toISOString(),
    inference: 'Packaged production Electron application, native bridge and real TensorRT-RTX model.',
    microphone: 'Twelve Windows SAPI generated speech fixtures through a test-only MediaStreamDestination; no user microphone captured.',
    expectedMinimumDurationSeconds: manifest.minimumDurationSeconds,
    expectedMinimumContextRollovers: manifest.minimumContextRollovers,
    turns: [], memory: [], errors: [],
  };
  const application = await _electron.launch({executablePath: executable, cwd: path.dirname(executable), args: [], env, timeout: 45000});
  let page;
  let lastProgress = 0;
  let readyAt = 0;

  async function snapshot() {
    return page.evaluate(() => {
      const test = window.conversationSoak;
      const now = Date.now();
      const activeSources = test.sources.filter(source => !source.stopped && !source.ended && source.context.currentTime >= source.when && source.context.currentTime < source.when + source.duration && source.rms > .0001);
      return {
        now, events: test.events.filter(event => event.type !== 'audio'), replies: Object.values(test.replies),
        errors: test.events.filter(event => event.type === 'error'),
        rollovers: test.events.filter(event => event.type === 'context_rolled'),
        activeSources: activeSources.map(source => ({id: source.id, epoch: source.epoch, at: source.at})),
        maxPlaybackLeadSeconds: test.maxPlaybackLeadSeconds,
        pendingSources: test.sources.filter(source => !source.stopped && !source.ended).length,
        pendingEpochs: [...new Set(test.sources.filter(source => !source.stopped && !source.ended).map(source => source.epoch))],
        fixtureTimings: test.fixture.starts,
        inputTrackStates: test.fixture.destination.stream.getTracks().map(track => track.readyState),
      };
    });
  }

  async function checkedSnapshot() {
    const state = await snapshot();
    assert.equal(state.errors.length, 0, JSON.stringify(state.errors));
    assert(state.maxPlaybackLeadSeconds < 2, `Playback lead grew to ${state.maxPlaybackLeadSeconds.toFixed(3)} seconds.`);
    if (readyAt && Date.now() - lastProgress >= 15000) {
      lastProgress = Date.now();
      console.log(JSON.stringify({elapsedSeconds: Math.round((Date.now() - readyAt) / 1000), completedTurns: receipt.turns.length, contextRollovers: state.rollovers.length, maxPlaybackLeadSeconds: state.maxPlaybackLeadSeconds, lastReply: state.replies.at(-1)?.text}));
      receipt.memory.push({at: Date.now(), processes: await application.evaluate(({app}) => app.getAppMetrics().map(process => ({pid: process.pid, type: process.type, memory: process.memory}))).catch(() => [])});
    }
    return state;
  }

  async function waitUntil(predicate, timeoutMs, description, intervalMs = 50) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const state = await checkedSnapshot();
      const result = predicate(state);
      if (result) return {state, result};
      await page.waitForTimeout(intervalMs);
    }
    throw new Error(`Timed out waiting for ${description}.`);
  }

  async function waitTo(targetAt) {
    while (Date.now() < targetAt) {
      await checkedSnapshot();
      await page.waitForTimeout(Math.min(1000, targetAt - Date.now()));
    }
  }

  async function playFixture(index) {
    return page.evaluate(async index => {
      const test = window.conversationSoak;
      const source = test.fixture.context.createBufferSource();
      source.buffer = test.fixture.buffers[index];
      source.connect(test.fixture.destination);
      if (index === 1) {
        // Start the spoken interruption while a chunk is still audible. A
        // fixed 100 ms injection delay can outlive an entire 80 ms chunk.
        // Observe actual overlap below, with the same 1.5 s outcome deadline.
        const deadline = Date.now() + 15000;
        while (!test.sources.some(playing => !playing.stopped && !playing.ended &&
          playing.rms > .0001 && playing.context.currentTime >= playing.when &&
          playing.when + playing.duration - playing.context.currentTime >= .03)) {
          if (Date.now() >= deadline) throw new Error('No audible chunk available for immediate spoken interruption.');
          await new Promise(resolve => setTimeout(resolve, 5));
        }
      }
      const injectionLeadSeconds = index === 1 ? 0 : .1;
      const startAt = Date.now() + injectionLeadSeconds * 1000;
      const record = {index, startAt, durationSeconds: source.buffer.duration, endAt: startAt + source.buffer.duration * 1000};
      test.fixture.starts.push(record);
      const when = test.fixture.context.currentTime + injectionLeadSeconds;
      const observeStart = setInterval(() => {
        if (test.fixture.context.currentTime < when) return;
        clearInterval(observeStart);
        record.observedStartAt = Date.now();
        record.playbackAtStart = test.sources.filter(playing => !playing.stopped && !playing.ended && playing.context.currentTime >= playing.when && playing.context.currentTime < playing.when + playing.duration && playing.rms > .0001).map(playing => ({id: playing.id, epoch: playing.epoch}));
      }, 5);
      source.start(when);
      return record;
    }, index);
  }

  function newReplies(state, timing) {
    return state.replies.filter(reply => reply.startedAt >= timing.startAt);
  }

  async function scoreTurn(index, timing, options = {}) {
    const turn = manifest.turns[index];
    const {state, result: reply} = await waitUntil(current => newReplies(current, timing).find(reply =>
      (options.allowPartial || reply.final) && containsExpected(reply.text, turn.expectedAny)),
    40000, `${turn.id} answer matching ${turn.expectedAny.join(' or ')}`);
    assertNoOldStory(reply.text, turn);
    const user = state.events.filter(event => event.type === 'transcript' && event.role === 'user' && event.final && event.at >= timing.startAt);
    assert(user.length > 0, `${turn.id} must have an actual final recognized input transcript.`);
    const record = {id: turn.id, prompt: turn.text, expectedAny: turn.expectedAny, timing, reply: {...reply}, userTranscripts: user, contextRolloversBeforeReply: state.rollovers.filter(event => event.at <= reply.updatedAt).length, interrupted: Boolean(options.allowPartial)};
    receipt.turns.push(record);
    return record;
  }

  try {
    page = await application.firstWindow();
    page.on('pageerror', error => receipt.errors.push(error.message));
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(() => document.querySelector('#modelDetail').textContent.includes('.bundle'));
    receipt.config = await page.evaluate(() => window.voiceLab.getConfig());
    assert(fs.statSync(receipt.config.bundlePath).size > 1024 * 1024 * 1024);
    await application.evaluate(({ipcMain}) => {
      globalThis.soakInputAudit = [];
      ipcMain.on('voice:audio', (_event, packet) => {
        let energy = 0;
        for (const sample of packet.samples) energy += sample * sample;
        globalThis.soakInputAudit.push({at: Date.now(), count: packet.samples.length, sampleRate: packet.sampleRate, rms: Math.sqrt(energy / packet.samples.length)});
      });
    });
    await page.evaluate(async fixtures => {
      const test = {events: [], sources: [], unmatchedSources: [], replies: {}, maxPlaybackLeadSeconds: 0, fixture: {}};
      window.conversationSoak = test;
      const originalStart = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(when = 0, ...args) {
        if (this.context !== test.fixture.context) {
          const samples = this.buffer.getChannelData(0);
          let energy = 0;
          for (const sample of samples) energy += sample * sample;
          const record = {id: test.sources.length, at: Date.now(), when, contextTime: this.context.currentTime, context: this.context, duration: this.buffer.duration, rms: Math.sqrt(energy / samples.length), stopped: false, ended: false};
          test.sources.push(record);
          test.unmatchedSources.push(record);
          test.maxPlaybackLeadSeconds = Math.max(test.maxPlaybackLeadSeconds, when + record.duration - record.contextTime);
          const originalStop = this.stop.bind(this);
          this.stop = (...stopArgs) => { record.stopped = true; record.stoppedAt = Date.now(); return originalStop(...stopArgs); };
          this.addEventListener('ended', () => { record.ended = true; record.endedAt = Date.now(); });
        }
        return originalStart.call(this, when, ...args);
      };
      window.voiceLab.onEvent(event => {
        const record = {...event, at: Date.now()};
        if (event.type === 'audio') {
          record.sampleCount = event.samples.length;
          record.rms = Math.sqrt(event.samples.reduce((sum, sample) => sum + sample * sample, 0) / event.samples.length);
          delete record.samples;
          const source = test.unmatchedSources.shift();
          if (source) source.epoch = event.epoch;
        }
        if (event.type === 'transcript' && event.role === 'assistant') {
          const reply = test.replies[event.epoch] ||= {epoch: event.epoch, startedAt: record.at, text: '', final: false};
          reply.text = event.delta ? reply.text + event.text : event.text;
          reply.final = event.final;
          reply.updatedAt = record.at;
        }
        if (event.type !== 'metrics') test.events.push(record);
      });
      navigator.mediaDevices.getUserMedia = async () => {
        const context = new AudioContext({sampleRate: 48000, latencyHint: 'interactive'});
        const destination = context.createMediaStreamDestination();
        const silence = context.createConstantSource();
        silence.offset.value = 0;
        silence.connect(destination);
        silence.start();
        test.fixture = {context, destination, buffers: [], starts: []};
        for (const fixture of fixtures) {
          const bytes = Uint8Array.from(atob(fixture), character => character.charCodeAt(0));
          test.fixture.buffers.push(await context.decodeAudioData(bytes.buffer.slice(0)));
        }
        await context.resume();
        return destination.stream;
      };
    }, fixtures);

    receipt.connectRequestedAt = Date.now();
    await page.click('#connectButton');
    await page.waitForFunction(() => conversationSoak.events.some(event => event.type === 'state' && event.backend === 'trt_rtx') || conversationSoak.events.some(event => event.type === 'error'), null, {timeout: 180000});
    const initial = await checkedSnapshot();
    readyAt = initial.events.find(event => event.backend === 'trt_rtx').at;
    receipt.readyAt = readyAt;
    receipt.loadSeconds = (readyAt - receipt.connectRequestedAt) / 1000;

    // Idle cancellation is an idempotent control operation and must not poison
    // the next microphone request or terminate the live session.
    receipt.idleInterrupt = {requestedAt: Date.now(), result: await page.evaluate(() => window.voiceLab.interrupt())};
    const idleAck = await waitUntil(state => state.events.find(event => event.type === 'flush' && event.reason === 'interrupt' && event.at >= receipt.idleInterrupt.requestedAt), 10000, 'idle interrupt acknowledgement');
    receipt.idleInterrupt.ack = idleAck.result;
    assert(idleAck.state.inputTrackStates.every(state => state === 'live'));

    await waitTo(readyAt + 8000);
    const storyTiming = await playFixture(0);
    // A duplex interruption overlaps assistant playback, never two injected
    // microphone fixtures. The model may reply before a WAV's last silence.
    await waitTo(storyTiming.endAt);
    // Establish the requested story topic before interrupting it. Otherwise a
    // valid early Stop can truncate the narrative before its required keyword.
    await scoreTurn(0, storyTiming, {allowPartial: true});
    const speaking = await waitUntil(state => state.activeSources.length > 0 && state.activeSources.some(source => source.at >= storyTiming.startAt), 20000, 'audible bedtime reply before spoken interruption', 20);
    const oldSourceIds = speaking.state.activeSources.map(source => source.id);
    const changeTiming = await playFixture(1);
    const start = await waitUntil(state => state.fixtureTimings.find(timing => timing.index === 1 && timing.observedStartAt), 2000, 'actual start of spoken interruption', 10);
    const overlap = start.result.playbackAtStart;
    assert(overlap.length > 0, 'Assistant audio must actually be audible when the Stop fixture starts.');
    const oldEpoch = overlap[0].epoch;
    const stopped = await waitUntil(state => {
      const yielded = state.events.find(event => event.type === 'flush' && event.reason === 'yielded' && event.at >= changeTiming.startAt);
      const completed = state.replies.find(reply => reply.epoch === oldEpoch && reply.final);
      if (!state.pendingEpochs.includes(oldEpoch) && (yielded || completed)) return {mechanism: yielded ? 'native-yield-and-flush' : completed.updatedAt >= changeTiming.startAt ? 'native-eos-and-playback-drain' : 'already-finished-native-playback-drain', terminal: yielded || completed};
      return false;
    }, 1500, 'old audible reply stopping within 1.5 seconds of spoken interruption', 10);
    receipt.speechInterruption = {startedAt: changeTiming.startAt, observedStartAt: start.result.observedStartAt, oldSourceIds, oldEpoch, ...stopped.result, latencyMs: stopped.state.now - start.result.observedStartAt};
    assert(receipt.speechInterruption.latencyMs <= 1500, 'Spoken interruption must stop the old audio within 1.5 seconds.');
    const stalePlayback = await page.evaluate(epoch => conversationSoak.sources.filter(source => source.epoch === epoch && !source.stopped && !source.ended).map(source => source.id), receipt.turns[0].reply.epoch);
    assert.equal(stalePlayback.length, 0, 'Spoken barge-in must discard all remaining audio from the abandoned response.');
    await scoreTurn(1, changeTiming);

    // Later topics span real context age boundaries. No reconnect/reset or
    // finish_input is used; silence stays on the same microphone stream.
    for (let index = 2; index < manifest.turns.length; index++) {
      // First exercise quick distinct follow-ups in the same model context;
      // then spread remaining topics across several real age-based refreshes.
      const targetAt = index < 4
        ? Math.max(Date.now(), receipt.turns.at(-1).timing.endAt) + 3000
        : readyAt + (60 + (index - 2) * 45) * 1000;
      await waitTo(targetAt);
      const timing = await playFixture(index);
      if (index === 8) {
        await scoreTurn(index, timing, {allowPartial: true});
        await waitUntil(state => state.activeSources.some(source => source.at >= timing.startAt), 15000, 'audible reply for explicit Stop speaking', 20);
        const before = await snapshot();
        const requestedAt = Date.now();
        await page.click('#interruptButton');
        const after = await waitUntil(state => state.events.find(event => event.type === 'flush' && event.reason === 'interrupt' && event.at >= requestedAt), 10000, 'Stop speaking acknowledgement', 20);
        const sources = await page.evaluate(() => conversationSoak.sources.map(({context, ...source}) => source));
        const oldSources = sources.filter(source => before.activeSources.some(active => active.id === source.id));
        assert(oldSources.length > 0 && oldSources.every(source => source.stopped || source.ended), 'Stop speaking must stop all audible prior sources.');
        assert(after.state.inputTrackStates.every(state => state === 'live'), 'Stop speaking must keep microphone capture alive.');
        receipt.buttonInterruption = {requestedAt, acknowledgedAt: after.result.at, oldSources, captureStillLive: true};
      } else {
        await scoreTurn(index, timing);
      }
    }
    await waitTo(readyAt + manifest.minimumDurationSeconds * 1000);
    const final = await waitUntil(state => state.rollovers.length >= manifest.minimumContextRollovers, 60000, `${manifest.minimumContextRollovers} actual context rollovers`, 1000);
    receipt.contextRollovers = final.state.rollovers;
    receipt.maxPlaybackLeadSeconds = final.state.maxPlaybackLeadSeconds;
    receipt.durationSeconds = (Date.now() - readyAt) / 1000;
    for (let index = 1; index < receipt.turns.length; index++) {
      const turn = receipt.turns[index];
      const nextStart = receipt.turns[index + 1]?.timing.startAt || Date.now();
      for (const reply of final.state.replies.filter(reply => reply.startedAt >= turn.timing.startAt && reply.startedAt < nextStart)) {
        assertNoOldStory(reply.text, manifest.turns[index]);
      }
    }
    const afterRefresh = receipt.turns.filter(turn => turn.contextRolloversBeforeReply > 0);
    assert(afterRefresh.length >= 5, 'At least five different topic answers must succeed after context refresh.');
    assert.equal(receipt.turns.length, 12);
    assert(receipt.buttonInterruption && receipt.speechInterruption && receipt.idleInterrupt.ack);
    assert.equal(receipt.errors.length, 0, JSON.stringify(receipt.errors));
    receipt.passed = true;
  } catch (error) {
    receipt.failure = error.message;
    throw error;
  } finally {
    if (page && !page.isClosed()) {
      const captured = await page.evaluate(() => ({events: window.conversationSoak?.events, replies: window.conversationSoak ? Object.values(conversationSoak.replies) : [], fixtureTimings: window.conversationSoak?.fixture?.starts, outputScheduling: window.conversationSoak?.sources.map(({context, ...source}) => source), maxPlaybackLeadSeconds: window.conversationSoak?.maxPlaybackLeadSeconds, state: document.querySelector('#sessionState')?.textContent})).catch(() => null);
      Object.assign(receipt, captured || {});
      receipt.inputPackets = await application.evaluate(() => globalThis.soakInputAudit).catch(() => []);
      const invalid = receipt.inputPackets?.filter(packet => packet.sampleRate !== 16000 || packet.count !== 320) || [];
      receipt.invalidInputPacketCount = invalid.length;
      if (invalid.length) { receipt.passed = false; receipt.failure ||= 'Capture did not consistently supply320-sample16kHz packets.'; }
      receipt.inputArrival = (receipt.inputPackets || []).reduce((result, packet, index, packets) => {
        if (index) { const gapMs = packet.at - packets[index - 1].at; result.maxGapMs = Math.max(result.maxGapMs, gapMs); if (gapMs >= 80) result.gapsAtLeast80ms++; }
        return result;
      }, {maxGapMs: 0, gapsAtLeast80ms: 0});
      await page.screenshot({path: receiptPath.replace(/\.json$/, '.png')}).catch(() => {});
      await page.evaluate(async () => { await window.voiceLab.disconnect(); if (window.conversationSoak?.fixture?.context?.state !== 'closed') await conversationSoak.fixture.context.close(); }).catch(() => {});
    }
    await application.close();
    receipt.finishedAt = new Date().toISOString();
    fs.mkdirSync(path.dirname(receiptPath), {recursive: true});
    fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
    console.log(JSON.stringify({passed: receipt.passed, completedTurns: receipt.turns.length, contextRollovers: receipt.contextRollovers?.length, maxPlaybackLeadSeconds: receipt.maxPlaybackLeadSeconds, failure: receipt.failure, receipt: receiptPath}));
    if (!receipt.passed) process.exitCode = 1;
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
