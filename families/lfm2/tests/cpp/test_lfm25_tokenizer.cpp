/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/byte_level_decoder.h"
#include "families/lfm2/runtime/pretokenizer.h"
#include "families/lfm2/runtime/tokenizer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const bool require_token_ids = argc == 3 && std::string(argv[2]) == "--token-ids";
    if (argc != 2 && !require_token_ids)
        return 1;
    const char* tokenizer_path = std::getenv("TRTMC_LFM25_TOKENIZER_JSON");
    if (require_token_ids && (tokenizer_path == nullptr || *tokenizer_path == '\0')) {
        std::cout
            << "SKIP: set TRTMC_LFM25_TOKENIZER_JSON to the pinned checkpoint tokenizer.json\n";
        return 77;
    }
    std::ifstream fixture_file(argv[1]);
    const auto fixtures = nlohmann::json::parse(fixture_file);
    std::unique_ptr<trtmc::ITokenizer> tokenizer;
    if (tokenizer_path != nullptr && *tokenizer_path != '\0') {
        std::ifstream input(tokenizer_path);
        if (!input) {
            std::cerr << "Cannot open the configured checkpoint tokenizer\n";
            return 1;
        }
        const std::string json((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
        tokenizer = trtmc::CreateBpeTokenizer(json.data(), json.size(), false, true);
        tokenizer = trtmc::lfm2_wrap_pinned_pretokenizer(
            std::move(tokenizer), trtmc::lfm2_tokenizer_added_tokens(json.data(), json.size()));
    }
    int failures = 0;
    for (const auto& fixture : fixtures.at("cases")) {
        const auto text = fixture.at("text").get<std::string>();
        if (trtmc::lfm2_split_pretokens(text) !=
            fixture.at("pieces").get<std::vector<std::string>>()) {
            std::cerr << "Pretoken mismatch: " << fixture.at("name") << '\n';
            ++failures;
        }
        if (tokenizer) {
            auto actual = tokenizer->encode(text);
            if (actual != fixture.at("token_ids").get<std::vector<int32_t>>()) {
                std::cerr << "Token ID mismatch: " << fixture.at("name") << '\n';
                ++failures;
            }
        }
    }
    std::cout << fixtures.at("cases").size() << " pretoken cases checked";
    if (tokenizer)
        std::cout << "; " << fixtures.at("cases").size() << " full token-ID cases checked";
    std::cout << '\n';
    return failures;
}
