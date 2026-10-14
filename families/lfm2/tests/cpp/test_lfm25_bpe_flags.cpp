/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream input(argv[1]);
    const auto fixtures = nlohmann::json::parse(input);
    int failures = 0;
    for (const auto& fixture : fixtures.at("cases")) {
        const auto config = fixture.at("tokenizer").dump();
        const auto tokenizer = trtmc::CreateBpeTokenizer(config.data(), config.size(), false, true);
        const auto actual = tokenizer->encode(fixture.at("text").get<std::string>());
        if (actual != fixture.at("token_ids").get<std::vector<int32_t>>()) {
            std::cerr << "BPE flag mismatch: " << fixture.at("name") << '\n';
            ++failures;
        }
    }
    std::cout << fixtures.at("cases").size() << " original BPE flag/fallback cases checked\n";
    return failures;
}
