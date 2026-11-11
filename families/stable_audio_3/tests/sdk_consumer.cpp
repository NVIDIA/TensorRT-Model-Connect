/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the stable_audio_3 family.
 * Renders a short clip through text_to_audio with the public C++ wrapper and
 * checks the declared Config fields and the output format.
 *
 * Usage:
 *   test_stable_audio_3_sdk_cpp <bundle_path> <runtime_root>
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
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
        auto task = model.task<trtmc::TextToAudio>();

        std::set<std::string> fields;
        for (const auto& field : task.config_fields())
            fields.insert(std::string(field.name));
        if (fields != std::set<std::string>{"duration", "steps", "seed"})
            throw std::runtime_error("text_to_audio: unexpected config fields");

        const trtmc::Config config{
            {"duration", 3.5}, {"steps", std::int64_t{4}}, {"seed", std::int64_t{5}}};
        auto result =
            task.run(trtmc::TextToAudioRequest{"A solo cello playing a slow melody"}, config);
        if (result.sample_rate() != 44100 || result.channels() != 2)
            throw std::runtime_error("text_to_audio: expected 44.1 kHz stereo");
        if (result.frame_count() != static_cast<std::uint64_t>(3.5 * 44100))
            throw std::runtime_error("text_to_audio: wrong frame count");
        for (const float value : result.samples())
            if (!std::isfinite(value) || value < -1.0F || value > 1.0F)
                throw std::runtime_error("text_to_audio: sample out of range");
        std::cout << "text_to_audio: frames=" << result.frame_count()
                  << " sample_rate=" << result.sample_rate() << " channels=" << result.channels()
                  << "\n";
    } catch (const std::exception& error) {
        std::cerr << "stable_audio_3 C++ SDK consumer error: " << error.what() << "\n";
        return 1;
    }
    std::cout << "stable_audio_3 C++ SDK consumer: all tasks passed.\n";
    return 0;
}
