/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/ernie4_5_moe/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
int regressions() {
    using nlohmann::json;
    const auto fixture = json::parse(R"({
      "model":{"type":"BPE","vocab":{"a":0,"Ġ":1,"b":2,"aĠ":3,"aĠb":4,"Ġa":5,"Ã":6,"©":7},"merges":[["a","Ġ"],["aĠ","b"],["Ġ","a"]]},
      "normalizer":{"type":"NFC"},
      "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":false},
      "decoder":{"type":"ByteLevel"},"added_tokens":[]
    })");
    auto make = [](const json& config) {
        const auto raw = config.dump();
        return trtmc::CreateBpeTokenizer(raw.data(), raw.size(), false);
    };
    const auto tokenizer = make(fixture);
    if (tokenizer->decode({6}) != "\xEF\xBF\xBD" || tokenizer->decode({6, 7}) != "é" ||
        tokenizer->decode({6, 0}) != "\xEF\xBF\xBD"
                                     "a")
        throw std::runtime_error(
            "generated token prefixes must decode as valid UTF-8 without breaking adjacent bytes");
    if (tokenizer->encode("a b") != std::vector<int32_t>{4})
        throw std::runtime_error("disabled byte-level regex must preserve cross-space merges");
    auto prefix = fixture;
    prefix["pre_tokenizer"]["add_prefix_space"] = true;
    if (make(prefix)->encode("a") != std::vector<int32_t>{5})
        throw std::runtime_error("direct ByteLevel prefix policy was ignored");
    auto regex = fixture;
    regex["pre_tokenizer"]["use_regex"] = true;
    if (make(regex)->encode("a b") != std::vector<int32_t>{0, 1, 2})
        throw std::runtime_error("enabled ByteLevel regex must retain its word boundaries");
    make(regex)->encode(std::string("\xF0", 1));
    make(regex)->encode(std::string("\xF0\x9F", 2));
    make(regex)->encode(std::string("\xF0\x9F\x98", 3));
    auto missing = fixture;
    missing["added_tokens"] = json::array({{{"id", 8}}});
    bool rejected = false;
    try {
        make(missing);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("incomplete added-token records must raise a load error");
    std::cout << "Tokenizer regressions passed\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1)
        return regressions();
    if (argc != 3)
        return 2;
    std::ifstream file(argv[1]);
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    const auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), true);
    std::cout << nlohmann::json(tokenizer->encode(argv[2])).dump() << '\n';
}
