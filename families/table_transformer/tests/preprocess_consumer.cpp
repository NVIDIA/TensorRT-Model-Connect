/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Expose the actual native preprocessing boundary to the independent PIL oracle.
#include "families/table_transformer/runtime/image_preprocess_seam.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        const int height = std::stoi(argv[3]), width = std::stoi(argv[4]);
        if (height <= 0 || width <= 0 || height > 10000 || width > 10000)
            throw std::invalid_argument("invalid test image dimensions");
        std::vector<float> input(static_cast<std::size_t>(height) * width * 3);
        std::ifstream source(argv[1], std::ios::binary);
        source.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        if (!source || source.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("invalid test input size");
        trtmc::TableTransformerPreprocessConfig config;
        config.dynamic_image_shapes = true;
        const auto result =
            trtmc::preprocess_table_transformer_image(input.data(), height, width, config);
        std::ofstream destination(argv[2], std::ios::binary);
        destination.write(reinterpret_cast<const char*>(result.data()),
                          result.size() * sizeof(float));
        if (!destination)
            throw std::runtime_error("unable to write preprocessing result");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
