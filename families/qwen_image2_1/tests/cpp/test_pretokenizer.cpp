/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The native tokenizer must split text exactly where the reference tokenizer does.
//
// tests/data/boundary_tokenizer.json has the checkpoint's pre-tokenizer, one token per byte and
// an end-of-word suffix: byte b encodes as id b inside a pre-token and as 256 + b at its end, so
// the ids returned by encode() spell out the pieces. tests/data/pretokenizer_cases.json holds the
// pieces the Hugging Face library produces with that tokenizer; test_pretokenizer_reference.py
// checks them against the library.
//
//   test_qwen_image2_1_pretokenizer BOUNDARY_TOKENIZER_JSON CASES_JSON
#include "families/qwen_image2_1/runtime/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kEndOfPiece = 256;
constexpr std::int32_t kFirstAddedToken = 512;

std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// The pre-tokens and added tokens that ids of the boundary tokenizer spell out.
std::vector<std::string> pieces(const std::vector<std::int32_t>& ids,
                                const std::vector<std::string>& added_tokens) {
    std::vector<std::string> result;
    std::string current;
    for (const std::int32_t id : ids) {
        if (id < 0)
            throw std::runtime_error("the tokenizer returned a negative id");
        if (id >= kFirstAddedToken) {
            const auto index = static_cast<std::size_t>(id - kFirstAddedToken);
            if (!current.empty() || index >= added_tokens.size())
                throw std::runtime_error("unexpected added token id " + std::to_string(id));
            result.push_back(added_tokens[index]);
            continue;
        }
        current.push_back(static_cast<char>(id % kEndOfPiece));
        if (id >= kEndOfPiece) {
            result.push_back(current);
            current.clear();
        }
    }
    if (!current.empty())
        throw std::runtime_error("the last piece has no end marker");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " BOUNDARY_TOKENIZER_JSON CASES_JSON\n";
        return 2;
    }
    try {
        const std::string tokenizer_json = read_file(argv[1]);
        const auto definition = nlohmann::json::parse(tokenizer_json);
        // The model that would read these ids has one row per token of the fixture.
        const auto token_count =
            definition.at("model").at("vocab").size() + definition.at("added_tokens").size();
        const auto tokenizer =
            trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false,
                                      static_cast<std::int32_t>(token_count));
        if (!tokenizer)
            throw std::runtime_error("the boundary tokenizer is not a BPE tokenizer");
        std::vector<std::string> added_tokens;
        for (const auto& token : definition.at("added_tokens"))
            added_tokens.push_back(token.at("content").get<std::string>());

        const auto document = nlohmann::json::parse(read_file(argv[2]));
        std::size_t checked = 0;
        std::size_t failed = 0;
        for (const auto& group : document.at("groups")) {
            for (const auto& item : group.at("cases")) {
                const auto text = item.at("text").get<std::string>();
                const auto expected = item.at("pieces").get<std::vector<std::string>>();
                const auto actual = pieces(tokenizer->encode(text), added_tokens);
                ++checked;
                if (actual == expected)
                    continue;
                ++failed;
                std::cerr << "FAILED [" << group.at("group").get<std::string>() << "] text "
                          << nlohmann::json(text).dump() << "\n  reference "
                          << nlohmann::json(expected).dump() << "\n  native    "
                          << nlohmann::json(actual).dump() << '\n';
            }
        }
        if (checked == 0)
            throw std::runtime_error("the case file holds no cases");
        std::cout << checked << " texts split, " << failed << " differ from the reference\n";
        return failed == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
