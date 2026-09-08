/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const test = require('node:test');
const assert = require('node:assert');

const trtmc = require('./build/Release/trtmc_node.node');

test('TensorRT-Model-Connect Node.js bindings API structure', async (t) => {
    await t.test('load() returns an object with a generate function', () => {
        const pipe = trtmc.load('dummy_model.bundle');
        assert.ok(pipe, 'Pipeline object should be returned');
        assert.strictEqual(typeof pipe.generate, 'function', 'Pipeline object should have a generate method');
    });

    await t.test('generate() returns the expected output structure', () => {
        const pipe = trtmc.load('dummy_model.bundle');
        const result = pipe.generate("Hello", { max_new_tokens: 10 });

        assert.ok(result, 'Result should be returned');
        assert.strictEqual(typeof result.text, 'string', 'Result should have a text property of type string');
        assert.ok(Array.isArray(result.token_ids), 'Result should have token_ids array');
        assert.strictEqual(typeof result.prefill_ms, 'number', 'Result should have prefill_ms number');
        assert.strictEqual(typeof result.decode_ms, 'number', 'Result should have decode_ms number');
        assert.strictEqual(typeof result.setup_ms, 'number', 'Result should have setup_ms number');
    });
});
