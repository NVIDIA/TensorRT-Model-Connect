/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/perception.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 8) return 2;
    trtmc::LoadOptions options; options.runtime_root = argv[2];
    const uint32_t height = std::stoul(argv[6]), width = std::stoul(argv[7]);
    std::vector<uint8_t> pixels(static_cast<size_t>(height) * width * 3), states(static_cast<size_t>(height) * width);
    std::ifstream rgb(argv[3], std::ios::binary), trimap(argv[4], std::ios::binary);
    rgb.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    trimap.read(reinterpret_cast<char*>(states.data()), states.size());
    if (!rgb || !trimap) return 2;
    const auto result = trtmc::Model::load(argv[1], options).task<trtmc::ImageTrimapToAlpha>().run(
        {trtmc::ImageInput({pixels.data(), pixels.size()}, height, width), {{states.data(), states.size()}, height, width}});
    const auto alpha = result.view().alpha;
    std::ofstream output(argv[5], std::ios::binary);
    output.write(reinterpret_cast<const char*>(alpha.data), alpha.count * sizeof(float));
    std::cout << nlohmann::json{{"rows", alpha.rows}, {"columns", alpha.columns}, {"count", alpha.count}}.dump() << '\n';
}
