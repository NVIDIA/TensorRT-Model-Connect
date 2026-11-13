/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pika_h3_flex/runtime/random.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    nlohmann::json input; std::cin >> input;
    trtmc::pika_h3_flex::Random random(input.at("seed").get<uint32_t>());
    std::ofstream out(argv[1], std::ios::binary);
    for (const auto& count : input.at("counts")) {
        const auto values = random.normal(count.get<std::size_t>());
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }
    return !out;
}
