/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visual_frontend.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> load(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Cannot open probe input");
    auto bytes = input.tellg();
    if (bytes < 0 || bytes % sizeof(float))
        throw std::runtime_error("Invalid probe input size");
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("Cannot read input");
    return values;
}
template <class T>
void emit(const std::vector<T>& values) {
    std::cout.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
int main(int argc, char** argv) {
    using namespace trtmc::nemotron_omni;
    try {
        if (argc < 4)
            throw std::runtime_error("Usage: visual_probe mode input dimensions...");
        const std::string mode = argv[1];
        auto number = [&](int index) {
            if (index >= argc)
                throw std::runtime_error("Missing argument");
            return std::stoi(argv[index]);
        };
        if (mode == "resize")
            emit(resize_rgb(load(argv[2]), number(3), number(4), number(5), number(6)));
        else if (mode == "positions")
            emit(visual_positions(load(argv[2]), number(3), number(4), number(5), number(6)));
        else if (mode == "shuffle")
            emit(shuffle_indices(number(3), number(4)));
        else if (mode == "evs") {
            if (argc != 6)
                throw std::runtime_error("Missing pruning rate");
            emit(video_retained_indices(load(argv[2]), number(3), number(4), std::stod(argv[5])));
        } else if (mode == "patch") {
            auto input = load(argv[2]);
            const int count = number(6);
            if (count < 1 || input.size() % count)
                throw std::runtime_error("Invalid input frames");
            const auto frame_size = input.size() / count;
            std::vector<std::vector<float>> frames;
            for (int index = 0; index < count; ++index)
                frames.emplace_back(input.begin() + index * frame_size,
                                    input.begin() + (index + 1) * frame_size);
            emit(patch_pixels(frames, number(3), number(4), number(5)));
        } else
            throw std::runtime_error("Unknown probe operation");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
