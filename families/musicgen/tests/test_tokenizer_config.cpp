/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/musicgen/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using nlohmann::json;

auto load(const json& config) {
    const auto text = config.dump();
    return trtmc::musicgen::CreateUnigramTokenizer(text.data(), text.size(), false);
}

void reject(const json& config) {
    try {
        (void)load(config);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("unsupported tokenizer configuration was accepted");
}

int main() {
    const json base = {
        {"model",
         {{"type", "Unigram"},
          {"unk_id", 0},
          {"vocab", json::array({json::array({"<unk>", -10.0}), json::array({"▁hello", -1.0}),
                                 json::array({"▁world", -1.0})})}}},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "WhitespaceSplit"}},
            {{"type", "Metaspace"},
             {"replacement", "▁"},
             {"prepend_scheme", "always"},
             {"split", true}}}}}},
        {"added_tokens",
         {{{"id", 3},
           {"content", "<special>"},
           {"lstrip", false},
           {"rstrip", false},
           {"normalized", false},
           {"single_word", false}}}}};
    if (load(base)->encode(" hello\tworld<special>hello ") != std::vector<int32_t>{1, 2, 3, 1})
        throw std::runtime_error("supported tokenizer configuration changed token IDs");
    for (const char* type : {"Replace", "Strip", "NFKC", "Lowercase", "Prepend", "Unknown"}) {
        auto config = base;
        config["normalizer"] = {{"type", type}};
        reject(config);
    }
    for (const json& normalizer :
         {json{{"type", "Sequence"}, {"normalizers", json::array()}},
          json{{"type", "Sequence"}, {"normalizers", {{{"type", "Strip"}}}}}}) {
        auto config = base;
        config["normalizer"] = normalizer;
        reject(config);
    }
    for (const json& pretokenizer :
         {json(nullptr), json{{"type", "ByteLevel"}}, base["pre_tokenizer"]["pretokenizers"][1]}) {
        auto config = base;
        config["pre_tokenizer"] = pretokenizer;
        reject(config);
    }
    for (const json& option :
         {json{{"add_prefix_space", false}}, json{{"prepend_scheme", "never"}},
          json{{"prepend_scheme", "first"}}, json{{"split", false}}, json{{"replacement", "_"}}}) {
        auto config = base;
        config["pre_tokenizer"]["pretokenizers"][1].update(option);
        reject(config);
    }
    for (const char* flag : {"lstrip", "rstrip", "normalized", "single_word"}) {
        auto config = base;
        config["added_tokens"][0][flag] = true;
        reject(config);
    }
    auto config = base;
    config["pre_tokenizer"]["pretokenizers"][0]["type"] = "Whitespace";
    reject(config);
    std::cout << "ALL PASSED\n";
}
