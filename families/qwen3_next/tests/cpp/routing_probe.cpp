/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/routing.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: qwen3_next_routing_probe PROBABILITIES_F32 TOP_K");
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        const std::streamoff bytes = input.tellg();
        if (!input || bytes < 0 || bytes % sizeof(float) != 0 ||
            bytes / sizeof(float) > std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("invalid router probability file");
        std::vector<float> probabilities(static_cast<std::size_t>(bytes) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(probabilities.data()), bytes);
        if (!input)
            throw std::runtime_error("cannot read router probabilities");
        const auto selected = trtmc::qwen3_next::select_experts(
            probabilities.data(), static_cast<std::int32_t>(probabilities.size()),
            std::stoi(argv[2]));
        std::vector<float> values;
        for (const auto index : selected)
            values.push_back(probabilities[index]);
        std::cout << nlohmann::json{{"indices", selected}, {"values", values}} << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
