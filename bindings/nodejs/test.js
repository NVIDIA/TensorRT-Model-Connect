/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const trtmc = require(process.env.TRTMC_NODE_ADDON || './build/Release/trtmc_node.node');
const runtime = process.env.TRTMC_TEST_RUNTIME_ROOT;
assert.ok(runtime, 'TRTMC_TEST_RUNTIME_ROOT must contain the built dataset and API fixture DSOs');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trtmc-node-'));
test.after(() => fs.rmSync(root, { recursive: true, force: true }));

function bundle(task, family = 'dataset_fixture') {
    const header = Buffer.from(JSON.stringify({
        format: 1, family, task, backend: 'fake',
        sections: { 'engine.plan': { offset: 0, length: 4 } },
    }));
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(header.length));
    const file = path.join(root, `${family}-${task}.bundle`);
    fs.writeFileSync(file, Buffer.concat([Buffer.from('BUNDLE\x01\x00'), length, header, Buffer.from('PLAN')]));
    return file;
}

function controls(result) {
    assert.equal(typeof result.text, 'string');
    assert.deepEqual(result.token_ids, [100, 101]);
    assert.equal(result.setup_ms, 1);
    assert.equal(result.prefill_ms, 2);
    assert.equal(result.decode_ms, 4);
    return JSON.parse(result.text.slice(result.text.lastIndexOf('\n') + 1));
}

test('semantic continuation uses family defaults and forwards repeated explicit controls', () => {
    const pipe = trtmc.load(bundle('text_continuation'), runtime);
    const defaults = controls(pipe.generate('First'));
    assert.equal(defaults.max_new_tokens, 5);
    assert.equal(defaults.temperature, 0.75);
    assert.equal(defaults.top_k, 7);
    const partial = controls(pipe.generate('Partial', { max_new_tokens: 9 }));
    assert.equal(partial.max_new_tokens, 9);
    assert.equal(partial.temperature, 0.75);
    assert.equal(partial.top_k, 7);
    for (const prompt of ['Second', 'Third']) {
        const result = pipe.generate(prompt, { max_new_tokens: 9, temperature: 0.5, top_k: 2 });
        assert.ok(result.text.startsWith(prompt));
        const config = controls(result);
        assert.equal(config.max_new_tokens, 9);
        assert.equal(config.temperature, 0.5);
        assert.equal(config.top_k, 2);
    }
});

test('legacy generation retains its declared loader and controls', () => {
    const pipe = trtmc.load(bundle('text_generation'), runtime);
    const defaults = controls(pipe.generate('Legacy defaults'));
    assert.equal(defaults.max_new_tokens, 128);
    assert.equal(defaults.top_k, 1);
    const config = controls(pipe.generate('Legacy', { max_new_tokens: 9, top_k: 2 }));
    assert.equal(config.max_new_tokens, 9);
    assert.equal(config.top_k, 2);
    assert.equal(Object.hasOwn(config, 'temperature'), false);
});

test('unsupported semantic Task fails at load without trying the old interface', () => {
    assert.throws(() => trtmc.load(bundle('disabled', 'api_fixture'), runtime));
});

test('SDK family rejection propagates without executing the available legacy method', () => {
    const pipe = trtmc.load(bundle('text_continuation'), runtime);
    assert.throws(() => pipe.generate('Too large', { max_new_tokens: 20001 }), /fixture token limit exceeded/);
    assert.ok(pipe.generate('After rejection').text.startsWith('After rejection'));
});

test('model-only fixture runs without legacy inheritance or undeclared implicit controls', () => {
    const pipe = trtmc.load(bundle('text_continuation', 'api_fixture'), runtime);
    const result = pipe.generate('SDK', { max_new_tokens: 9, temperature: 0.5 });
    assert.equal(result.text, 'SDK!|eos');
    assert.equal(result.prefill_ms, 0.5);
    assert.equal(result.decode_ms, 9);
    assert.throws(() => pipe.generate('Unsupported', { top_k: 1 }), /does not declare config: top_k/);
});

test('SDK numeric controls retain precision and reject lossy or nonnumeric input', () => {
    const pipe = trtmc.load(bundle('text_continuation'), runtime);
    const value = 0.123456789012345;
    assert.equal(controls(pipe.generate('Precise', { temperature: value })).temperature, value);
    for (const temperature of [NaN, Infinity, -Infinity, '0.5', null, true]) {
        assert.throws(() => pipe.generate('Invalid', { temperature }), /must be (finite|a number)/);
    }
    for (const name of ['max_new_tokens', 'top_k']) {
        for (const value of [1.5, 2 ** 53, NaN, Infinity, '2', null, true]) {
            assert.throws(() => pipe.generate('Invalid', { [name]: value }), /must be (a safe integer|finite|a number)/);
        }
    }
    assert.equal(controls(pipe.generate('Valid after rejection')).max_new_tokens, 5);
});
