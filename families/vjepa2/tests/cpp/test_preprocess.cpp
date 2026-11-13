/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/preprocess.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main() {
    nlohmann::json input;
    std::cin >> input;
    const int32_t h = input.at("height"), w = input.at("width");
    const int32_t frames = input.value("frames", 1);
    if (frames <= 0)
        return 2;
    const auto frame_size = static_cast<std::size_t>(h) * w * 3;
    std::vector<float> pixels(frames * frame_size);
    std::ifstream file(input.at("source").get<std::string>(), std::ios::binary);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!file)
        return 2;
    const auto geometry = trtmc::vjepa2::image_geometry(h, w, 292, 256);
    std::ofstream result(input.at("output").get<std::string>(), std::ios::binary);
    for (int32_t frame = 0; frame < frames; ++frame) {
        const auto output =
            trtmc::vjepa2::preprocess(pixels.data() + frame * frame_size, h, w, 292, 256);
        result.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    }
    std::cout << nlohmann::json{{"height", geometry.height},
                                {"width", geometry.width},
                                {"top", geometry.top},
                                {"left", geometry.left}}
                     .dump()
              << '\n';
}
