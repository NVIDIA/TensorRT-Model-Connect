/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the stable_audio family: one short text_to_audio
 * request through the public C++ wrapper (trtmc/trtmc.hpp + trtmc/audio.hpp),
 * after checking that the Task declares the family's config fields.
 *
 * Usage:
 *   test_stable_audio_sdk_cpp <bundle_path> <runtime_root>
 */

#include <algorithm>
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
        auto task = model.task<trtmc::TextToAudio>();

        const auto fields = task.config_fields();
        for (const char* name : {"negative_prompt", "audio_start_in_s", "audio_end_in_s",
                                 "num_inference_steps", "guidance_scale", "seed"}) {
            const bool declared = std::any_of(fields.begin(), fields.end(), [&](const auto& field) {
                return field.name == name;
            });
            if (!declared)
                throw std::runtime_error(std::string("text_to_audio does not declare ") + name);
        }

        // Guidance off: one denoiser branch over the model's whole window.
        const trtmc::Config config{
            {"num_inference_steps", 4}, {"guidance_scale", 1.0}, {"seed", 3}};
        auto result = task.run(trtmc::TextToAudioRequest{"A short click."}, config);
        std::cout << "text_to_audio: samples=" << result.samples().size()
                  << " sample_rate=" << result.sample_rate() << " channels=" << result.channels()
                  << "\n";
        if (result.sample_rate() != 44100 || result.channels() != 2 || result.samples().empty() ||
            result.samples().size() % 2 != 0)
            throw std::runtime_error("text_to_audio: expected stereo frames at 44100 Hz");
        std::cout << "stable_audio C++ SDK consumer: all tasks passed\n";
    } catch (const std::exception& error) {
        std::cerr << "stable_audio C++ SDK consumer failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
