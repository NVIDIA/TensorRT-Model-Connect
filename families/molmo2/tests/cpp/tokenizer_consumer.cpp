/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/molmo2/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc < 2)
        return 2;
    try {
        std::ifstream file(argv[1]);
        const std::string text((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        auto tokenizer = trtmc::molmo2::CreateBpeTokenizer(text.data(), text.size(), false);
        nlohmann::json output = nlohmann::json::array();
        for (int i = 2; i < argc; ++i)
            output.push_back(tokenizer->encode(argv[i]));
        std::cout << output.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
