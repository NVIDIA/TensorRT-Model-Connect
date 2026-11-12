/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the gpt2 family.
 * Exercises text_continuation through the public C++ convenience wrapper
 * (trtmc/trtmc.hpp).
 *
 * Usage:
 *   sdk_consumer_gpt2_cpp <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_GPT2_PROMPT   text prompt (default: "The quick brown fox jumps over the lazy dog")
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/trtmc.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root>\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];

    const char* env_prompt = std::getenv("TRTMC_GPT2_PROMPT");
    const std::string prompt =
        env_prompt ? env_prompt : "The quick brown fox jumps over the lazy dog";

    try {
        // ── load model ────────────────────────────────────────────────────
        trtmc::LoadOptions opts;
        opts.runtime_root = runtime_root;
        auto model = trtmc::Model::load(bundle_path, opts);

        if (!model.supports<trtmc::TextContinuation>()) {
            throw std::runtime_error("model does not support text_continuation");
        }

        // ── text_continuation ─────────────────────────────────────────────
        {
            auto task = model.task<trtmc::TextContinuation>();
            trtmc::TextContinuationRequest req{prompt};
            auto result = task.run(req);

            if (result.text().empty() && result.token_ids().empty()) {
                throw std::runtime_error("text_continuation: result is empty");
            }

            std::cout << "text_continuation: generated " << result.token_ids().size()
                      << " tokens: " << result.text() << "\n";
        }

        // ── invalid prompt rejection test ──────────────────────────────────
        {
            auto task = model.task<trtmc::TextContinuation>();
            bool caught = false;
            try {
                trtmc::TextContinuationRequest bad_req{std::string("")};
                (void)task.run(bad_req);
            } catch (const std::exception&) {
                caught = true;
            }
            if (!caught) {
                throw std::runtime_error("expected empty prompt to be rejected");
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "gpt2 C++ SDK consumer error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "gpt2 C++ SDK consumer: all tasks passed.\n";
    return 0;
}
