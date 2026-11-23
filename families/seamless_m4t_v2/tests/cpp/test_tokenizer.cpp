/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/seamless_m4t_v2/runtime/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    std::ifstream file(argv[1]);
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    std::ifstream config(std::filesystem::path(argv[1]).parent_path() / "config.json");
    const auto cfg = nlohmann::json::parse(config);
    const auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), false,
                                                     cfg.at("vocab_size").get<std::size_t>());
    std::cout << nlohmann::json(tokenizer->encode(argv[2])).dump() << '\n';
}
