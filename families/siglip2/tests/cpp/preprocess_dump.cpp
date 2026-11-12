/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs the runtime's NaFlex preprocessing on one raw 8-bit RGB image and writes what the
// vision engine would receive, so the family test can compare it with the reference
// processor value for value.
//
// usage: siglip2_preprocess_dump <rgb.u8> <height> <width> <patch> <budget> <grid>
//                                <mean r g b> <std r g b> <out.bin>
// out.bin: int32 grid_height, int32 grid_width, float32 pixel_values, float32 taps.

#include "families/siglip2/runtime/naflex_preprocess.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 14)
            throw std::invalid_argument("expected 13 arguments, see the source header");
        const auto height = std::stoi(argv[2]);
        const auto width = std::stoi(argv[3]);
        trtmc::Siglip2PreprocessConfig config;
        config.patch_size = std::stoi(argv[4]);
        config.max_num_patches = std::stoi(argv[5]);
        config.position_grid = std::stoi(argv[6]);
        config.image_mean = {std::stof(argv[7]), std::stof(argv[8]), std::stof(argv[9])};
        config.image_std = {std::stof(argv[10]), std::stof(argv[11]), std::stof(argv[12])};

        std::ifstream input(argv[1], std::ios::binary);
        const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),
                                               std::istreambuf_iterator<char>());
        if (bytes.size() != static_cast<std::size_t>(height) * width * 3)
            throw std::invalid_argument("raw image size does not match height x width x 3");
        std::vector<float> rgb(bytes.size());
        for (std::size_t index = 0; index < bytes.size(); ++index)
            rgb[index] = static_cast<float>(bytes[index]) / 255.0F;

        const auto patches = trtmc::preprocess_siglip2_image(rgb.data(), height, width, config);
        std::ofstream output(argv[13], std::ios::binary);
        output.write(reinterpret_cast<const char*>(&patches.grid_height), sizeof(std::int32_t));
        output.write(reinterpret_cast<const char*>(&patches.grid_width), sizeof(std::int32_t));
        output.write(reinterpret_cast<const char*>(patches.pixel_values.data()),
                     static_cast<std::streamsize>(patches.pixel_values.size() * sizeof(float)));
        output.write(reinterpret_cast<const char*>(patches.position_weights.data()),
                     static_cast<std::streamsize>(patches.position_weights.size() * sizeof(float)));
        if (!output)
            throw std::runtime_error("could not write the output file");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
