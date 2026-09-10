'use strict';
const path = require('node:path');
const crypto = require('node:crypto');

// Bounded lifecycle diagnostics. Audio and transcript text never enter the log;
// session-local keyed hashes allow repeated replies to be identified.
function createDiagnostics(directory, fs = require('node:fs'), maxBytes = 2 * 1024 * 1024) {
  const file = path.join(directory, 'voice-lab-runtime.jsonl');
  const previous = path.join(directory, 'voice-lab-runtime.previous.jsonl');
  const key = crypto.randomBytes(32);
  const run = crypto.randomUUID();
  let size = 0, enabled = true;
  try {
    fs.mkdirSync(directory, {recursive: true});
    size = fs.statSync(file, {throwIfNoEntry: false})?.size || 0;
  } catch { enabled = false; }
  function record(event) {
    if (!enabled || !event || ['audio', 'metrics'].includes(event.type)) return;
    if (event.type === 'transcript' && event.final !== true) return;
    const row = {at: new Date().toISOString(), run, type: event.type};
    for (const field of ['state', 'kind', 'reason', 'interruptStatus', 'role']) {
      if (typeof event[field] === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(event[field])) row[field] = event[field];
    }
    for (const field of ['epoch', 'sequence']) if (Number.isSafeInteger(event[field])) row[field] = event[field];
    if (event.type === 'transcript' && typeof event.text === 'string') {
      row.characters = event.text.length;
      row.fingerprint = crypto.createHmac('sha256', key).update(event.text.toLowerCase().replace(/\s+/g, ' ').trim()).digest('hex').slice(0, 24);
    }
    if (event.type === 'context_rolled') {
      for (const match of String(event.message || '').matchAll(/\b(segment|reason|prior_steps|memory_tokens|memory_policy|rebuild_ms)=([a-zA-Z0-9_-]+)/g)) row[match[1]] = match[2].slice(0, 64);
    }
    const line = JSON.stringify(row) + '\n';
    try {
      if (size + Buffer.byteLength(line) > maxBytes) {
        fs.rmSync(previous, {force: true});
        if (fs.existsSync(file)) fs.renameSync(file, previous);
        size = 0;
      }
      fs.appendFileSync(file, line);
      size += Buffer.byteLength(line);
    } catch { enabled = false; }
  }
  record({type: 'application_started'});
  return {record};
}

module.exports = {createDiagnostics};
