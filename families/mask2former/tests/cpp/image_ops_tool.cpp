/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs the family's image operations on raw files so that tests/test_image_ops.py can compare them
// with Pillow, the reference image processor and torch.
//
//   resize     IN.u8  H W OUT_H OUT_W OUT.u8
//   preprocess IN.u8  H W OUT_H OUT_W RESCALE MEAN0 MEAN1 MEAN2 STD0 STD1 STD2 OUT.f32
//   argmax     IN.f32 CLASSES SCORE_H SCORE_W H W OUT.i32

#include "families/mask2former/runtime/image_ops.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::int32_t integer(const char* text) {
    std::size_t consumed = 0;
    const auto value = std::stol(text, &consumed);
    if (text[consumed] != '\0' || value <= 0 || value > 1 << 20)
        throw std::invalid_argument(std::string("invalid size: ") + text);
    return static_cast<std::int32_t>(value);
}

float number(const char* text) {
    std::size_t consumed = 0;
    const auto value = std::stof(text, &consumed);
    if (text[consumed] != '\0')
        throw std::invalid_argument(std::string("invalid number: ") + text);
    return value;
}

template <class T>
std::vector<T> read(const char* path, std::size_t count) {
    std::vector<T> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error(std::string("input size does not match the stated shape: ") +
                                 path);
    return values;
}

template <class T>
void write(const char* path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!file)
        throw std::runtime_error(std::string("cannot write ") + path);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "resize" && argc == 8) {
            const auto height = integer(argv[3]), width = integer(argv[4]);
            const auto input =
                read<std::uint8_t>(argv[2], static_cast<std::size_t>(height) * width * 3);
            write(argv[7], trtmc::mask2former::resize_bilinear_rgb8(
                               input.data(), height, width, integer(argv[5]), integer(argv[6])));
            return 0;
        }
        if (mode == "preprocess" && argc == 15) {
            const auto height = integer(argv[3]), width = integer(argv[4]);
            const auto input =
                read<std::uint8_t>(argv[2], static_cast<std::size_t>(height) * width * 3);
            trtmc::mask2former::PreprocessConfig config;
            config.height = integer(argv[5]);
            config.width = integer(argv[6]);
            config.rescale_factor = std::stod(argv[7]);
            config.mean = {number(argv[8]), number(argv[9]), number(argv[10])};
            config.std = {number(argv[11]), number(argv[12]), number(argv[13])};
            write(argv[14], trtmc::mask2former::preprocess(input.data(), height, width, config));
            return 0;
        }
        if (mode == "argmax" && argc == 9) {
            const auto classes = integer(argv[3]);
            const auto score_height = integer(argv[4]), score_width = integer(argv[5]);
            const auto scores = read<float>(argv[2], static_cast<std::size_t>(classes) *
                                                         score_height * score_width);
            write(argv[8], trtmc::mask2former::resize_argmax(scores.data(), classes, score_height,
                                                             score_width, integer(argv[6]),
                                                             integer(argv[7])));
            return 0;
        }
        std::cerr << "usage: see the header of image_ops_tool.cpp\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
