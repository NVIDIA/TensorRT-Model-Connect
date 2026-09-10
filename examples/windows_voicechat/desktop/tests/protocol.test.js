const {test} = require('node:test');
const assert = require('node:assert/strict');
const {normalizeEvent, decodeAudio, validateInputAudio} = require('../protocol');

test('PCM bridge audio decodes exactly and carries timing identity', () => {
  const bytes = Buffer.alloc(12); [0, .5, -.25].forEach((x, i) => bytes.writeFloatLE(x, i * 4));
  const [packet] = normalizeEvent({type: 'event', kind: 'agent_audio', audio: bytes.toString('base64'), sampleRate: 48000, epoch: 3, sequence: 9});
  assert.deepEqual(packet.samples, [0, .5, -.25]); assert.equal(packet.epoch, 3); assert.equal(packet.sampleRate, 48000);
});
test('yield, cancellation, and reset invalidate scheduled audio', () => {
  for (const kind of ['yielded', 'cancelled', 'reset']) {
    const flush = normalizeEvent({type: 'event', kind})[0];
    assert.equal(flush.type, 'flush');
    assert.equal(flush.reason, kind, 'Renderer needs to distinguish an output interruption from an input reset.');
  }
});
test('invalid audio cannot enter native session', () => {
  assert.throws(() => validateInputAudio({samples: [0], sampleRate: 48000}));
  assert.throws(() => validateInputAudio({samples: [NaN], sampleRate: 16000}));
  assert.throws(() => validateInputAudio({samples: new Array(16001).fill(0), sampleRate: 16000}));
  assert.deepEqual(validateInputAudio({samples: new Float32Array([0, .5]), sampleRate: 16000}).samples, [0, .5]);
  assert.throws(() => decodeAudio('AA=='));
  assert.throws(() => decodeAudio('!!!!'));
  assert.throws(() => validateInputAudio({samples: [1.001], sampleRate: 16000}));
});

test('ready handshake requires the requested model, backend and audio contract', () => {
  const ready = {type: 'ready', backend: 'trt_rtx', family: 'nemotron_voicechat', protocolVersion: 1, inputSampleRate: 16000, outputSampleRate: 48000};
  assert.equal(normalizeEvent(ready)[0].state, 'listening');
  for (const incompatible of [{backend: 'trt'}, {family: 'other'}, {protocolVersion: 2}, {inputSampleRate: 48000}, {outputSampleRate: 16000}]) {
    assert.throws(() => normalizeEvent({...ready, ...incompatible}));
  }
});

test('fatal inference failures flush queued speech, nonfatal command errors preserve it', () => {
  const fatal = normalizeEvent({type: 'event', kind: 'error', text: 'GPU execution failed', epoch: 2, sequence: 4});
  assert.equal(fatal[0].type, 'flush');
  assert.equal(fatal[1].fatal, true);
  assert.equal(normalizeEvent({type: 'error', message: 'Runtime failed', fatal: true})[0].type, 'flush');
  const warning = normalizeEvent({type: 'error', message: 'Unknown command', fatal: false});
  assert.equal(warning.length, 1);
  assert.equal(warning[0].type, 'error');
});

test('corrupt output metadata cannot silently produce incorrect playback', () => {
  const event = {type: 'event', kind: 'agent_audio', audio: Buffer.alloc(16).toString('base64'), sampleRate: 48000, sampleCount: 4, encoding: 'f32le'};
  assert.equal(normalizeEvent(event)[0].samples.length, 4);
  assert.throws(() => normalizeEvent({...event, sampleCount: 5}));
  assert.throws(() => normalizeEvent({...event, sampleRate: 16000}));
  assert.throws(() => normalizeEvent({...event, encoding: 's16le'}));
});
test('transcript roles and finality survive normalization', () => {
  assert.deepEqual(normalizeEvent({type: 'event', kind: 'user_transcript', text: 'Hello', isFinal: true, epoch: 1, sequence: 2})[0], {type: 'transcript', role: 'user', text: 'Hello', final: true, delta: false, epoch: 1, sequence: 2});
  assert.equal(normalizeEvent({type: 'event', kind: 'agent_text', text: 'Hi', isFinal: false})[0].delta, true);
});
