/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the bark family.
 * Exercises text_to_audio through the public C++ convenience wrapper
 * (trtmc/trtmc.hpp + trtmc/audio.hpp).
 *
 * Usage:
 *   sdk_consumer_bark_cpp <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_BARK_PROMPT   text prompt (default: "Hello from Bark.")
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <trtmc/trtmc.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root>\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];

    const char* env_prompt = std::getenv("TRTMC_BARK_PROMPT");
    const std::string prompt = env_prompt ? env_prompt : "Hello from Bark.";

    try {
        // ── load model ────────────────────────────────────────────────────
        trtmc::LoadOptions opts;
        opts.runtime_root = runtime_root;
        auto model = trtmc::load_task(bundle_path, opts);

        // ── text_to_audio ─────────────────────────────────────────────────
        {
            auto& audio_task = model->get<trtmc::TextToAudio>();
            trtmc::TextToAudioRequest req{prompt};
            auto result = audio_task.run(req);

            if (result.samples().empty())
                throw std::runtime_error("text_to_audio: result has no samples");
            if (result.sample_rate() != 24000)
                throw std::runtime_error("text_to_audio: expected sample_rate == 24000");
            if (result.channels() != 1)
                throw std::runtime_error("text_to_audio: expected channels == 1");

            std::cout << "text_to_audio: samples=" << result.samples().size()
                      << " sample_rate=" << result.sample_rate()
                      << " channels=" << result.channels() << " duration_s="
                      << static_cast<double>(result.frame_count()) / result.sample_rate() << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "bark C++ SDK consumer error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "bark C++ SDK consumer: all tasks passed.\n";
    return 0;
}
