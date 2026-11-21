/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/image_processor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("usage: probe H W MIN MAX TILES SPLIT OUTPUT_PREFIX");
        trtmc::ImageResult image;
        image.height = std::stoi(argv[1]);
        image.width = std::stoi(argv[2]);
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x) {
                image.pixels.push_back((x * 13 + y * 7) % 256);
                image.pixels.push_back((x * 3 + y * 17) % 256);
                image.pixels.push_back((x + y * 31) % 256);
            }
        trtmc::lfm2_vl::ImageOptions options;
        options.min_tokens = std::stoi(argv[3]);
        options.max_tokens = std::stoi(argv[4]);
        options.max_tiles = std::stoi(argv[5]);
        options.split = std::stoi(argv[6]) != 0;
        const auto result = trtmc::lfm2_vl::process_image(image, options);
        std::cout << result.placeholders << '\n';
        std::size_t index = 0;
        for (const auto& tile : result.tiles) {
            const std::string path = std::string(argv[7]) + std::to_string(index++);
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(tile.pixels.data()),
                         tile.pixels.size() * sizeof(float));
            const std::vector<float> position_table(16 * 16 * 1152, 0.0F);
            const auto prepared =
                trtmc::lfm2_vl::prepare_vision_inputs(tile, options, position_table, 1152, true);
            std::ofstream padded(path + ".padded", std::ios::binary);
            padded.write(reinterpret_cast<const char*>(prepared.pixels.data()),
                         prepared.pixels.size() * sizeof(float));
            std::ofstream mask(path + ".mask", std::ios::binary);
            mask.write(reinterpret_cast<const char*>(prepared.mask.data()),
                       prepared.mask.size() * sizeof(std::int32_t));
            std::cout << tile.patch_rows << ' ' << tile.patch_cols << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
