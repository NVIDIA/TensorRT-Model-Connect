/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cosmos3_edge/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Added = std::vector<std::pair<std::string, int32_t>>;
std::string fixture(const Added& added) {
    auto j = nlohmann::json::parse(R"({
      "model":{"type":"BPE","dropout":null,"vocab":{},"merges":[]},
      "normalizer":null,
      "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        {"type":"Split","behavior":"Isolated","pattern":{"Regex":".+"}},
        {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
      "decoder":{"type":"ByteLevel"},"added_tokens":[]})");
    for (int value = 33; value <= 126; ++value)
        j["model"]["vocab"][std::string(1, static_cast<char>(value))] = value;
    for (const auto& [text, id] : added)
        j["added_tokens"].push_back({{"content", text}, {"id", id}, {"special", true}});
    return j.dump();
}

// Independent simple oracle: inspect every token at each character position.
std::vector<int32_t> reference(std::string_view text, const Added& added) {
    std::vector<int32_t> result;
    size_t at = 0;
    while (at < text.size()) {
        size_t length = 0;
        int32_t token = 0;
        for (const auto& [candidate, id] : added) {
            if (candidate.size() > length && text.substr(at, candidate.size()) == candidate) {
                length = candidate.size();
                token = id;
            }
        }
        if (length) {
            result.push_back(token);
            at += length;
        } else {
            result.push_back(static_cast<unsigned char>(text[at++]));
        }
    }
    return result;
}
} // namespace

int main() {
    try {
        const Added added{{"ab", 1000},     {"abc", 1001}, {"bc", 1002},    {"<m>", 1003},
                          {"<m><m>", 1004}, {"x", 1005},   {"absent", 1006}};
        const trtmc::cosmos3_edge::Tokenizer tokenizer(fixture(added));
        auto check = [&](const std::string& input) {
            if (tokenizer.encode(input) != reference(input, added))
                throw std::runtime_error("Added-token earliest/longest matching changed");
        };
        for (const auto* input : {"", "abcabc", "zabcbc", "xx", "<m><m><m>", "a<m>b"})
            check(input);
        std::mt19937 random(7);
        const std::vector<std::string> chunks{"abc", "bc", "z", "x", "<m>", "a", "b"};
        for (int trial = 0; trial < 100; ++trial) {
            std::string input;
            for (int i = 0; i < 100; ++i)
                input += chunks[random() % chunks.size()];
            check(input);
        }
        std::string media;
        for (int i = 0; i < 4096; ++i)
            media += "<m>";
        check(media + "abc");
        const trtmc::cosmos3_edge::Tokenizer ordinary(fixture({}));
        if (ordinary.encode("abc") != std::vector<int32_t>{97, 98, 99})
            throw std::runtime_error("Ordinary byte tokenization changed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
