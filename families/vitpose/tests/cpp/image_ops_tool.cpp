/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs the family's image operations on raw files so that tests/test_image_ops.py can compare them
// with the reference processor:
//
//   preprocess IN.u8 H W BOX_X BOX_Y BOX_W BOX_H OUT_H OUT_W PADDING RESCALE M0 M1 M2 S0 S1 S2
//              OUT.u8 OUT.f32
//       writes the 8-bit HWC crop and the CHW float32 pixel values, and prints the crop centre
//       and scale as one JSON line.
//   decode     IN.f32 KEYPOINTS H W CENTER_X CENTER_Y SCALE_X SCALE_Y KERNEL OUT.f32
//       writes x, y, score and located (1 or 0) per keypoint as float32.
//
// Numbers are read with strtod, so a Python repr round-trips exactly.

#include "families/vitpose/runtime/image_ops.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::int32_t size(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > (1L << 20))
        throw std::invalid_argument(std::string("invalid size: ") + text);
    return static_cast<std::int32_t>(value);
}

double number(const char* text) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0')
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

int preprocess(char** argv) {
    const auto height = size(argv[3]);
    const auto width = size(argv[4]);
    const auto pixels = read<std::uint8_t>(argv[2], static_cast<std::size_t>(height) * width * 3U);
    const trtmc::vitpose::PersonBox box{number(argv[5]), number(argv[6]), number(argv[7]),
                                        number(argv[8])};
    trtmc::vitpose::PreprocessConfig config;
    config.height = size(argv[9]);
    config.width = size(argv[10]);
    config.padding_factor = number(argv[11]);
    config.rescale_factor = number(argv[12]);
    for (int channel = 0; channel < 3; ++channel) {
        config.mean[static_cast<std::size_t>(channel)] =
            static_cast<float>(number(argv[13 + channel]));
        config.std[static_cast<std::size_t>(channel)] =
            static_cast<float>(number(argv[16 + channel]));
    }
    const auto geometry = trtmc::vitpose::crop_geometry(box, config);
    const auto crop = trtmc::vitpose::warp_crop(pixels.data(), height, width, geometry, config);
    write(argv[19], crop);
    write(argv[20], trtmc::vitpose::normalize_crop(crop, config));
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"center\": ["
              << geometry.center[0] << ", " << geometry.center[1] << "], \"scale\": ["
              << geometry.scale[0] << ", " << geometry.scale[1] << "]}\n";
    return 0;
}

int decode(char** argv) {
    const auto keypoints = size(argv[3]);
    const auto height = size(argv[4]);
    const auto width = size(argv[5]);
    const auto heatmaps = read<float>(argv[2], static_cast<std::size_t>(keypoints) *
                                                   static_cast<std::size_t>(height) * width);
    trtmc::vitpose::CropGeometry geometry;
    geometry.center = {static_cast<float>(number(argv[6])), static_cast<float>(number(argv[7]))};
    geometry.scale = {static_cast<float>(number(argv[8])), static_cast<float>(number(argv[9]))};
    const auto kernel = size(argv[10]);
    const auto decoded = trtmc::vitpose::decode_heatmaps(heatmaps.data(), keypoints, height, width,
                                                         geometry, kernel);
    std::vector<float> values;
    for (const auto& keypoint : decoded) {
        values.insert(values.end(),
                      {keypoint.x, keypoint.y, keypoint.score, keypoint.located ? 1.0F : 0.0F});
    }
    write(argv[11], values);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "preprocess" && argc == 21)
            return preprocess(argv);
        if (command == "decode" && argc == 12)
            return decode(argv);
        std::cerr << "usage: see the comment at the top of image_ops_tool.cpp\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "vitpose_image_ops_tool: " << error.what() << '\n';
        return 1;
    }
}
