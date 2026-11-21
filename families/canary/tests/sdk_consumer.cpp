/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the canary family.
 * Exercises speech_transcription through the public C++ convenience wrappers
 * (trtmc/trtmc.hpp + trtmc/audio.hpp).
 *
 * Usage:
 *   sdk_consumer_canary_cpp <bundle_path> <runtime_root>
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <trtmc/trtmc.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root>\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];

    try {
        trtmc::LoadOptions opts;
        opts.runtime_root = runtime_root;
        auto model = trtmc::Model::load(bundle_path, opts);

        auto task = model.task<trtmc::SpeechTranscription>();

        std::vector<float> pcm(16000, 0.0F);
        trtmc::SpeechTranscriptionRequest req;
        req.audio.samples = {pcm.data(), pcm.size()};
        req.audio.sample_rate = 16000;
        req.audio.channels = 1;

        auto result = task.run(req);
        std::cout << "speech_transcription text size=" << result.text().size() << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "canary C++ SDK consumer error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "canary C++ SDK consumer: passed.\n";
    return 0;
}
