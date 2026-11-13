/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 8)
        return 2;
    trtmc::LoadOptions options;
    options.runtime_root = argv[2];
    const uint32_t frames = std::stoul(argv[5]), h = std::stoul(argv[6]), w = std::stoul(argv[7]);
    const auto frame_size = static_cast<std::size_t>(h) * w * 3;
    std::vector<float> pixels(frames * frame_size);
    std::ifstream input(argv[3], std::ios::binary);
    input.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!input)
        return 2;
    trtmc::VideoToTokenFeaturesRequest request;
    for (uint32_t i = 0; i < frames; ++i) {
        request.video.frames.emplace_back(
            trtmc::Span<const float>{pixels.data() + i * frame_size, frame_size}, h, w);
        request.video.timestamps_seconds.push_back(i / 30.0);
    }
    const auto result =
        trtmc::Model::load(argv[1], options).task<trtmc::VideoToTokenFeatures>().run(request);
    const auto features = result.features();
    std::ofstream output(argv[4], std::ios::binary);
    output.write(reinterpret_cast<const char*>(features.values.data()),
                 features.values.size() * sizeof(float));
    auto token = [](const trtmc::VideoFeatureToken& value) {
        return nlohmann::json{{"role", value.role},
                              {"grid", {value.grid_frame, value.grid_row, value.grid_column}},
                              {"span", {value.frame_begin, value.frame_end}},
                              {"box", {value.x_min, value.y_min, value.x_max, value.y_max}}};
    };
    std::cout << nlohmann::json{{"rows", features.rows},
                                {"columns", features.columns},
                                {"grid",
                                 {result.grid_frames(), result.grid_rows(), result.grid_columns()}},
                                {"source_frames", result.source_frame_count()},
                                {"first", token(result.tokens()[0])},
                                {"last", token(result.tokens()[result.tokens().size() - 1])}}
                     .dump()
              << '\n';
}
