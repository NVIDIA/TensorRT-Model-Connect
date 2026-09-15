/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const trtmc = require('./build/Release/trtmc_node.node');
const path = require('path');
const fs = require('fs');

console.log("TRT-Model-Connect Node.js Bindings Loaded Successfully.");

// Mock bundle path
const bundlePath = process.argv[2] || "dummy_model.bundle";

try {
    console.log(`Loading bundle: ${bundlePath}...`);
    const pipe = trtmc.load(bundlePath);
    console.log("Model loaded successfully!");

    console.log("Running inference...");
    const result = pipe.generate("Hello, how are you?", {
        max_new_tokens: 50,
        temperature: 0.7
    });

    console.log("Inference Result:");
    console.log(`  Text: ${result.text}`);
    console.log(`  Prefill Time: ${result.prefill_ms} ms`);
    console.log(`  Decode Time: ${result.decode_ms} ms`);
    console.log(`  Tokens: [${result.token_ids.join(', ')}]`);

} catch (e) {
    console.error("Error during inference:", e);
}
// DCO Remediation
