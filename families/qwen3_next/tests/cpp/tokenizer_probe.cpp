/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: qwen3_next_tokenizer_probe TOKENIZER TEXT");
        std::ifstream input(argv[1]);
        const std::string data{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
        if (!input)
            throw std::runtime_error("cannot read tokenizer");
        const auto tokenizer =
            trtmc::qwen3_next::CreateBpeTokenizer(data.data(), data.size(), false);
        const auto ids = tokenizer->encode(argv[2]);
        std::cout << nlohmann::json{{"ids", ids}, {"text", tokenizer->decode(ids)}} << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
