/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/flint/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: flint_tokenizer_probe TOKENIZER_JSON");
        std::ifstream stream(argv[1]);
        if (!stream)
            throw std::runtime_error("cannot read Flint test tokenizer");
        const std::string document((std::istreambuf_iterator<char>(stream)), {});
        auto tokenizer = trtmc::CreateBpeTokenizer(document.data(), document.size(), false);
        nlohmann::json requests;
        std::cin >> requests;
        auto results = nlohmann::json::array();
        for (const auto& request : requests) {
            const auto ids = request.contains("ids")
                                 ? request.at("ids").get<std::vector<std::int32_t>>()
                                 : tokenizer->encode(request.at("text").get<std::string>());
            results.push_back({{"ids", ids}, {"text", tokenizer->decode(ids)}});
        }
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
