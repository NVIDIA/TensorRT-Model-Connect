/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <trtmc/audio.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToSpeech>();
        const trtmc::Config deterministic{{"temperature", 0.0}};
        const auto first = task.run({"Hello from Phonon.", "en"}, deterministic);
        const std::vector<float> saved(first.samples().begin(), first.samples().end());
        if (first.sample_rate() != 24000 || first.channels() != 1 || saved.size() != 46080)
            throw std::runtime_error("Phonon audio geometry changed");
        bool rejected = false;
        try {
            task.run({"Hello from Phonon.", "fr"}, deterministic);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Phonon accepted an unsupported language");
        for (const std::int64_t seed : {std::int64_t{-1}, std::int64_t{4294967296}}) {
            rejected = false;
            try {
                task.run({"Hello from Phonon.", "en"}, {{"temperature", 0.0}, {"seed", seed}});
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Phonon accepted a seed outside its 32-bit range");
        }
        const auto second = task.run({"Hello from Phonon.", {}}, deterministic);
        if (second.samples().size() != saved.size() ||
            !std::equal(saved.begin(), saved.end(), second.samples().begin()) ||
            !std::equal(saved.begin(), saved.end(), first.samples().begin()))
            throw std::runtime_error("Phonon request reuse or result ownership failed");
        for (const auto value : saved)
            if (!std::isfinite(value))
                throw std::runtime_error("Phonon returned nonfinite PCM");
        std::cout << "Phonon C++ SDK: deterministic reuse, language/seed rejection and retained "
                     "result passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
