/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exposes the family's native preprocessing and score resize to its Python
// parity test, which compares them with the official torchvision processor:
//
//   preprocess RGB_U8 HEIGHT WIDTH SHORTEST LONGEST MEAN0 MEAN1 MEAN2 STD0 STD1 STD2 RESCALE
//   OUT_F32
//     -> writes planar float32 pixel values, prints {"height":H,"width":W}
//   argmax SCORES_F32 CLASSES HEIGHT WIDTH TARGET_HEIGHT TARGET_WIDTH OUT_I32
//     -> writes int32 labels of the resized score maps

#include "families/oneformer/runtime/image_ops.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <class T>
std::vector<T> read_exactly(const std::string& path, std::size_t count) {
    std::vector<T> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("input file does not hold exactly the expected values: " + path);
    return values;
}

template <class T>
void write_all(const std::string& path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!file)
        throw std::runtime_error("cannot write " + path);
}

std::int32_t positive(const char* text) {
    const auto value = std::stol(text);
    if (value <= 0 || value > 1 << 24)
        throw std::invalid_argument(std::string("expected a positive size, got ") + text);
    return static_cast<std::int32_t>(value);
}

} // namespace

int main(int argc, char** argv) {
    using namespace trtmc::oneformer;
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "preprocess" && argc == 15) {
            const ImageSize source{positive(argv[3]), positive(argv[4])};
            const auto pixels = read_exactly<std::uint8_t>(
                argv[2], static_cast<std::size_t>(source.height) * source.width * 3);
            Normalization normalization;
            for (int channel = 0; channel < 3; ++channel) {
                normalization.mean[static_cast<std::size_t>(channel)] =
                    std::stof(argv[7 + channel]);
                normalization.std[static_cast<std::size_t>(channel)] =
                    std::stof(argv[10 + channel]);
            }
            normalization.rescale_factor = std::stod(argv[13]);
            const auto target = processed_size(source, positive(argv[5]), positive(argv[6]));
            const auto resized = resize_uint8_antialias(pixels.data(), source, target, 3);
            write_all(argv[14], normalize_planar(resized, target, normalization));
            std::cout << "{\"height\":" << target.height << ",\"width\":" << target.width << "}\n";
            return 0;
        }
        if (command == "argmax" && argc == 9) {
            const auto classes = positive(argv[3]);
            const ImageSize source{positive(argv[4]), positive(argv[5])};
            const ImageSize target{positive(argv[6]), positive(argv[7])};
            const auto scores = read_exactly<float>(argv[2], static_cast<std::size_t>(classes) *
                                                                 source.height * source.width);
            write_all(argv[8], resize_argmax(scores.data(), classes, source, target));
            return 0;
        }
        std::cerr << "usage: " << argv[0]
                  << " preprocess ... | argmax ... (see the source header)\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
