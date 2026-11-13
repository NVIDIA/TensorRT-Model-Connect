/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/audio.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    trtmc::LoadOptions options; options.runtime_root = argv[2];
    const uint64_t frames = std::stoull(argv[5]);
    std::vector<int32_t> features(frames * 8);
    std::ifstream input(argv[3], std::ios::binary);
    input.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(int32_t));
    if (!input) return 2;
    const auto result = trtmc::Model::load(argv[1], options).task<trtmc::AudioCodebookTokensToAudio>().run(
        {{{features.data(), features.size()}, 8, frames}});
    std::ofstream output(argv[4], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.samples().data()), result.samples().size() * sizeof(float));
    std::cout << nlohmann::json{{"samples", result.samples().size()}, {"rate", result.sample_rate()},
        {"channels", result.channels()}}.dump() << '\n';
}
