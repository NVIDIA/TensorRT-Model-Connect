// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const express = require('express');

let trtmc;
try {
    trtmc = require('tensorrt-model-connect-node/build/Release/trtmc_node.node');
} catch (e) {
    console.warn("[WARN] Native module not found. Using MOCK for demonstration.");
    trtmc = {
        load: (path) => ({
            generate: (prompt, config) => ({
                text: "This is a simulated AI response from the Node.js bindings via Express!",
                token_ids: [101, 202, 303],
                prefill_ms: 10.5,
                decode_ms: 30.1,
                setup_ms: 0.5
            })
        })
    };
}

const app = express();
app.use(express.json());

const BUNDLE_PATH = process.env.BUNDLE_PATH || "dummy_model.bundle";
const pipeline = trtmc.load(BUNDLE_PATH);
console.log(`Model bundle loaded from ${BUNDLE_PATH}`);

app.post('/api/generate', (req, res) => {
    const { prompt, max_new_tokens = 50, temperature = 0.7 } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: "Prompt is required" });
    }

    try {
        console.log(`Received generate request with prompt: "${prompt}"`);
        const start = Date.now();
        const result = pipeline.generate(prompt, { max_new_tokens, temperature });
        const latency = Date.now() - start;

        res.json({
            success: true,
            response: result.text,
            metrics: {
                prefill_ms: result.prefill_ms,
                decode_ms: result.decode_ms,
                setup_ms: result.setup_ms,
                total_latency_ms: latency
            }
        });
    } catch (e) {
        console.error("Inference Error:", e);
        res.status(500).json({ error: "Internal inference error", details: e.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`TensorRT-Model-Connect Node.js API server running on port ${PORT}`);
});
