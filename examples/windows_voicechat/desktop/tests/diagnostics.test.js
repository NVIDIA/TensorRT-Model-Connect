'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {createDiagnostics} = require('../diagnostics');

function memoryFs() {
  const files = new Map();
  return {files, mkdirSync() {},
    statSync(file) { return files.has(file) ? {size: Buffer.byteLength(files.get(file))} : undefined; },
    existsSync(file) { return files.has(file); },
    appendFileSync(file, text) { files.set(file, (files.get(file) || '') + text); },
    rmSync(file) { files.delete(file); },
    renameSync(from, to) { files.set(to, files.get(from)); files.delete(from); },
  };
}

test('diagnostics identify repeated replies without recording speech or transcript content', () => {
  const fs = memoryFs();
  const logger = createDiagnostics('logs', fs);
  logger.record({type: 'audio', samples: [0.125], audio: 'secret audio'});
  logger.record({type: 'transcript', role: 'user', text: 'private partial', final: false});
  logger.record({type: 'transcript', role: 'assistant', text: 'Private phrase', final: true});
  logger.record({type: 'transcript', role: 'assistant', text: ' private   PHRASE ', final: true});
  logger.record({type: 'context_rolled', message: 'segment=4 reason=repeated-response memory_tokens=0 memory_policy=latest-request-only rebuild_ms=12 private content'});
  const text = [...fs.files.values()].join('');
  assert(!/private|secret audio|samples/i.test(text));
  const rows = text.trim().split('\n').map(line => JSON.parse(line));
  const transcripts = rows.filter(row => row.type === 'transcript');
  assert.equal(transcripts.length, 2);
  assert.equal(transcripts[0].fingerprint, transcripts[1].fingerprint);
  assert.equal(rows.at(-1).reason, 'repeated-response');
  assert.equal(rows.at(-1).memory_tokens, '0');
  assert.equal(rows.at(-1).memory_policy, 'latest-request-only');
});

test('diagnostic history remains bounded to the current file and one previous file', () => {
  const fs = memoryFs();
  const logger = createDiagnostics('logs', fs, 512);
  for (let index = 0; index < 200; index++) logger.record({type: 'state', state: 'listening', epoch: index});
  assert.equal(fs.files.size, 2);
  for (const text of fs.files.values()) assert(Buffer.byteLength(text) <= 512);
  const latest = fs.files.get(path.join('logs', 'voice-lab-runtime.jsonl')).trim().split('\n').map(JSON.parse);
  assert.equal(latest.at(-1).epoch, 199);
});

test('unwritable diagnostics never stop the voice application', () => {
  const fs = memoryFs();
  fs.appendFileSync = () => { throw new Error('Disk full'); };
  const logger = createDiagnostics('logs', fs);
  assert.doesNotThrow(() => logger.record({type: 'state', state: 'listening'}));
});
