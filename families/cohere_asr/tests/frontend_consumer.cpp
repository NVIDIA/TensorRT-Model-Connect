/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere_asr/runtime/frontend.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static std::vector<float> read(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(float))
        throw std::runtime_error("invalid input float file");
    const auto bytes = file.tellg();
    std::vector<float> result(static_cast<std::size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file)
        throw std::runtime_error("cannot read input");
    return result;
}
int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        const auto audio = read(argv[1]), window = read(argv[2]), filters = read(argv[3]);
        const auto values =
            trtmc::cohere_asr::features(audio.data(), audio.size(), window, filters);
        std::ofstream file(argv[4], std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        return file ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
