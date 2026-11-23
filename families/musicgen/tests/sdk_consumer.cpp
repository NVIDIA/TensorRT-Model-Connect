/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <trtmc/audio.hpp>
#include <trtmc/trtmc.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToAudio>();
        const auto run = [&](const trtmc::Config& config) {
            auto result = task.run({"A mellow jazz piano with soft drums."}, config);
            if (result.sample_rate() != 32000 || result.channels() != 1 || result.samples().empty())
                throw std::runtime_error("wrong native audio format");
            std::vector<float> pcm(result.samples().begin(), result.samples().end());
            if (!std::all_of(pcm.begin(), pcm.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("non-finite native audio");
            return pcm;
        };
        trtmc::Config sampled{{"seed", std::int64_t{42}}};
        const auto first = run(sampled);
        // Reject an invalid request between calls on the same loaded Task.
        bool rejected = false;
        try {
            run(trtmc::Config{{"temperature", 0.0}});
        } catch (const trtmc::Error&) {
            rejected = true;
        }
        if (!rejected || first != run(sampled))
            throw std::runtime_error("request reset or seeded sampling reproducibility failed");
        if (first == run(trtmc::Config{{"seed", std::int64_t{43}}}))
            throw std::runtime_error("sampling seed did not affect the audio");
        rejected = false;
        try {
            task.run({std::string(10000, 'a')}, sampled);
        } catch (const trtmc::Error&) {
            rejected = true;
        }
        if (!rejected || first != run(sampled))
            throw std::runtime_error("prompt bounds or recovery after invalid prompt failed");
        std::cout << "MusicGen public SDK: sampling, repeated calls, invalid requests, and "
                     "recovery passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
