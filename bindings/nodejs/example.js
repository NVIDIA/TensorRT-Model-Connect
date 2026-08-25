// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

let trtmc;
try {
    trtmc = require('./build/Release/trtmc_node.node');
} catch (e) {
    console.warn("[WARN] Native module not found. Using MOCK for demonstration purposes.");
    trtmc = {
        load: (path) => ({
            generate: (prompt, config) => ({
                text: "This is a simulated AI response from the Node.js bindings!",
                token_ids: [101, 202, 303, 404],
                prefill_ms: 12.5,
                decode_ms: 45.2,
                setup_ms: 1.0
            })
        })
    };
}
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
