/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the vibevoice family: text_to_speech through the
 * public C++ convenience wrapper (trtmc/trtmc.hpp + trtmc/audio.hpp).
 *
 * Usage:
 *   test_vibevoice_sdk_cpp <bundle_path> <runtime_root>
 */

#include <cmath>
#include <cstdint>
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
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToSpeech>();
        bool seed_declared = false;
        for (const auto& field : task.config_fields())
            seed_declared = seed_declared || field.name == "seed";
        if (!seed_declared)
            throw std::runtime_error("text_to_speech does not declare its seed config");

        trtmc::Config config;
        config.add("seed", std::int64_t{7});
        config.add("max_new_tokens", std::int64_t{48});
        const auto result =
            task.run({"Speaker 1: Hello from the C plus plus SDK.", std::nullopt}, config);
        if (result.samples().empty() || result.sample_rate() != 24000 || result.channels() != 1)
            throw std::runtime_error("text_to_speech returned unexpected audio");
        if (result.samples().size() % 3200 != 0)
            throw std::runtime_error("text_to_speech audio is not a whole number of frames");
        for (const float sample : result.samples()) {
            if (!std::isfinite(sample))
                throw std::runtime_error("text_to_speech audio has a non-finite sample");
        }
        bool rejected = false;
        try {
            (void)task.run({"Hello.", std::string("en")}, config);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("text_to_speech accepted a language it cannot control");
        std::cout << "text_to_speech: samples=" << result.samples().size()
                  << " sample_rate=" << result.sample_rate() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "vibevoice C++ SDK consumer error: " << error.what() << "\n";
        return 1;
    }
    std::cout << "vibevoice C++ SDK consumer: all checks passed.\n";
    return 0;
}
