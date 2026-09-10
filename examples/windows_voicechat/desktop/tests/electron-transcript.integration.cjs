'use strict';

// Opt-in Electron regression: exercises real renderer/preload event delivery.
// No model or microphone is opened, and Chromium GPU rendering is disabled.
const {_electron} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

async function run() {
  const desktop = path.resolve(__dirname, '..');
  const testRoot = path.resolve(process.env.VOICE_LAB_TEST_ROOT || os.tmpdir());
  fs.mkdirSync(testRoot, {recursive: true});
  const workspace = fs.mkdtempSync(path.join(testRoot, 'voice-lab-transcript-test-'));
  const env = {...process.env, VOICE_LAB_WORKSPACE: workspace};
  delete env.ELECTRON_RUN_AS_NODE;
  const executablePath = path.resolve(process.env.ELECTRON_EXECUTABLE || (process.versions.electron ? process.execPath : require('electron')));
  const application = await _electron.launch({executablePath, cwd: path.dirname(executablePath), args: [desktop, '--disable-gpu'], env, timeout: 30000});
  try {
    const page = await application.firstWindow();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(() => document.querySelector('#modelDetail').textContent.includes('.bundle'));
    await page.evaluate(() => { document.querySelector('#transcriptEmpty').dataset.sentinel = 'original'; });
    const emit = events => application.evaluate(({BrowserWindow}, packets) => {
      const contents = BrowserWindow.getAllWindows()[0].webContents;
      for (const packet of packets) contents.send('voice:event', packet);
    }, events);
    const rows = () => page.locator('.transcript-entry, .transcript-notice').evaluateAll(elements => elements.map(element => ({notice: element.classList.contains('transcript-notice'), text: element.querySelector('.entry-text')?.textContent || element.textContent, partial: Boolean(element.querySelector('.entry-text.partial'))})));

    const history = [];
    for (let index = 0; index < 360; index += 1) {
      history.push({type: 'transcript', role: 'user', text: `Historical turn ${index}`, final: true});
      history.push({type: 'context_rolled', message: `Refresh ${index}`});
    }
    history.push({type: 'transcript', role: 'user', text: 'Current partial', final: false});
    await emit(history);
    await page.waitForFunction(() => document.querySelector('.transcript-entry:last-child .entry-text')?.textContent === 'Current partial');
    const retained = await rows();
    assert.equal(retained.length, 300, 'Speech and refresh notices must share one 300-row budget.');
    assert.equal(retained.filter(row => row.notice).length, 150, 'Refresh notices must participate in pruning.');
    assert(!retained.some(row => row.text === 'Historical turn 210'), 'Oldest history must be evicted.');
    assert(retained.some(row => row.text === 'Historical turn 211'));
    assert(retained.some(row => row.text === 'Historical turn 359'), 'Newest completed transcript must remain visible.');
    assert.deepEqual(retained.at(-1), {notice: false, text: 'Current partial', partial: true});
    await page.evaluate(() => { document.querySelector('.transcript-entry:last-child').dataset.retained = 'current'; });
    await emit([{type: 'transcript', role: 'user', text: 'Current partial completed.', final: true}]);
    await page.waitForFunction(() => document.querySelector('[data-retained="current"] .entry-text')?.textContent === 'Current partial completed.');
    assert.equal((await rows()).length, 300, 'Updating the current partial must reuse its visible row.');

    // An unusual long-lived partial may itself age out. Its future snapshot must
    // create a visible row instead of updating an evicted, detached DOM element.
    await page.click('#clearTranscript');
    await emit([{type: 'transcript', role: 'user', id: 'old-partial', text: 'Old pending turn', final: false}]);
    const speechOnly = Array.from({length: 310}, (_, index) => ({type: 'transcript', role: 'assistant', text: `Later turn ${index}`, final: true}));
    speechOnly.push({type: 'transcript', role: 'user', id: 'old-partial', text: 'Returned pending turn', final: false});
    await emit(speechOnly);
    await page.waitForFunction(() => document.querySelector('.transcript-entry:last-child .entry-text')?.textContent === 'Returned pending turn');
    const afterEviction = await rows();
    assert.equal(afterEviction.length, 300);
    assert.deepEqual(afterEviction.at(-1), {notice: false, text: 'Returned pending turn', partial: true});
    assert.equal(await page.locator('#transcriptEmpty').getAttribute('data-sentinel'), 'original', 'Pruning must preserve the empty-state element.');
    assert.deepEqual(errors, [], 'No renderer exceptions are allowed.');
    console.log(JSON.stringify({events: history.length + 1 + speechOnly.length + 1, visibleRows: afterEviction.length, combinedNoticeBudget: true, currentPartialRetained: true, evictedActiveEntryRecovered: true, emptyStatePreserved: true}, null, 2));
  } finally {
    await application.close();
    console.log(`Isolated test profile: ${workspace}`);
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
