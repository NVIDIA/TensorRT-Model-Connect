/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_frontend.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> load(const char* name) {
    std::ifstream file(name, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open probe input");
    const auto bytes = file.tellg();
    if (bytes < 0 || bytes % sizeof(float))
        throw std::runtime_error("Invalid probe input size");
    std::vector<float> result(static_cast<std::size_t>(bytes) / sizeof(float));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), bytes))
        throw std::runtime_error("Cannot read probe input");
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("Usage: probe PCM filters maximum_frames");
        const auto features =
            trtmc::nemotron_omni::audio_features(load(argv[1]), load(argv[2]), std::stoul(argv[3]));
        std::cout.write(reinterpret_cast<const char*>(features.values.data()),
                        static_cast<std::streamsize>(features.values.size() * sizeof(float)));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
