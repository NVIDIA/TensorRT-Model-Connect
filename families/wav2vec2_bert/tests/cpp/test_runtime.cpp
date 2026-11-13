/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    trtmc::LoadOptions options; options.runtime_root = argv[2];
    const uint64_t frames = std::stoull(argv[5]);
    std::vector<float> features(frames * 160);
    std::ifstream input(argv[3], std::ios::binary);
    input.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(float));
    if (!input) return 2;
    const auto result = trtmc::Model::load(argv[1], options).task<trtmc::SpectrogramToFrameFeatures>().run(
        {{{features.data(), features.size()}, frames, 160}});
    std::ofstream output(argv[4], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.features().values.data()), result.features().values.size() * sizeof(float));
    std::vector<uint64_t> mapping(result.input_frame_indices().begin(), result.input_frame_indices().end());
    std::cout << nlohmann::json{{"rows", result.features().rows}, {"columns", result.features().columns},
        {"source_frames", result.source_frame_count()}, {"input_frame_indices", mapping}}.dump() << '\n';
}
