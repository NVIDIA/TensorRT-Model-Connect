'use strict';

function decodeAudio(base64) {
  if (typeof base64 !== 'string' || base64.length > 8 * 1024 * 1024 || base64.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(base64)) throw new Error('Invalid audio packet');
  const bytes = Buffer.from(base64, 'base64');
  if (bytes.length % 4) throw new Error('Unaligned PCM packet');
  const samples = new Array(bytes.length / 4);
  for (let i = 0; i < samples.length; i++) {
    const value = bytes.readFloatLE(i * 4);
    if (!Number.isFinite(value)) throw new Error('Non-finite output audio');
    samples[i] = Math.max(-1, Math.min(1, value));
  }
  return samples;
}

function normalizeEvent(event) {
  if (!event || typeof event !== 'object' || Array.isArray(event) || typeof event.type !== 'string') throw new Error('Invalid native event');
  if (event.type === 'loading') return [{type: 'state', state: 'loading', message: 'Loading local TensorRT-RTX engines…'}];
  if (event.type === 'ready') {
    if (event.backend !== 'trt_rtx' || event.family !== 'nemotron_voicechat' || event.protocolVersion !== 1 || event.inputSampleRate !== 16000 || event.outputSampleRate !== 48000) throw new Error('The bridge must provide protocol 1 Nemotron VoiceChat on TensorRT-RTX with 16 kHz input and 48 kHz output.');
    return [{type: 'state', state: 'listening', backend: event.backend, inputSampleRate: event.inputSampleRate}];
  }
  if (event.type === 'stopped') return [{type: 'flush'}, {type: 'state', state: 'disconnected'}];
  if (event.type === 'error') return event.fatal === false ? [event] : [{type: 'flush'}, {...event, fatal: true}];
  if (event.type !== 'event') return [event];
  const common = {epoch: event.epoch, sequence: event.sequence};
  switch (event.kind) {
    case 'agent_audio': {
      if (event.sampleRate !== 48000 || (event.encoding !== undefined && event.encoding !== 'f32le')) throw new Error('Invalid native output audio format');
      const samples = decodeAudio(event.audio);
      if (event.sampleCount !== undefined && event.sampleCount !== samples.length) throw new Error('Native output sample count does not match PCM');
      return [{...common, type: 'audio', samples, sampleRate: event.sampleRate}];
    }
    case 'agent_text':
    case 'user_transcript': return [{...common, type: 'transcript', role: event.kind === 'agent_text' ? 'assistant' : 'user', text: event.text, final: event.isFinal, delta: event.kind === 'agent_text' && !event.isFinal}];
    case 'turn_started': return [{...common, type: 'state', state: 'thinking'}];
    case 'turn_finished': return [{...common, type: 'state', state: 'listening'}];
    case 'user_speech_started': return [{...common, type: 'state', state: 'listening'}];
    case 'yielded':
    case 'reset':
    case 'cancelled': return [{...common, type: 'flush', reason: event.kind}, {...common, type: 'state', state: 'listening'}];
    case 'context_rolled': return [{...common, type: 'context_rolled', message: event.text}];
    case 'error': return [{...common, type: 'flush'}, {...common, type: 'error', fatal: true, message: event.text || 'Native voice session failed.'}];
    default: return [{...common, type: 'lifecycle', kind: event.kind}];
  }
}

function validateInputAudio(packet) {
  if (!packet || packet.sampleRate !== 16000) throw new Error('Microphone audio must be 16 kHz mono.');
  const samples = Array.from(packet.samples || []);
  if (!samples.length || samples.length > 16000 || samples.some(x => !Number.isFinite(x) || Math.abs(x) > 1)) {
    throw new Error('Invalid microphone PCM.');
  }
  return {type: 'audio', sampleRate: 16000, samples};
}

module.exports = {decodeAudio, normalizeEvent, validateInputAudio};
