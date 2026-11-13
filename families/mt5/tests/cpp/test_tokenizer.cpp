/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/mt5/runtime/tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1]);
    std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto tokenizer = trtmc::CreateUnigramTokenizer(data.data(), data.size(), true);
    auto ids = tokenizer->encode(argv[2]);
    std::cout << nlohmann::json{{"ids", ids}, {"text", tokenizer->decode(ids)}} << '\n';
}
