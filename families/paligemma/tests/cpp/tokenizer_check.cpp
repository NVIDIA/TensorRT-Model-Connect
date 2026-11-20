/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compares the native PaliGemma prompt assembly and decoding, loaded from a
// built bundle, with ids and text the reference processor produced.
//
//   paligemma_tokenizer_check BUNDLE CASES.json
//
// CASES.json: {"prompts": [{"prefix", "images", "input_ids"}],
//              "decodes": [{"token_ids", "text"}]}
// Prints one JSON line with the case counts and any mismatches; exits non-zero on a mismatch.

#include "families/paligemma/runtime/paligemma_bpe.h"
#include "families/paligemma/runtime/prompt.h"
#include "trtmc/bundle.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: paligemma_tokenizer_check BUNDLE CASES.json\n";
        return 2;
    }
    try {
        using namespace trtmc::paligemma;
        const trtmc::BundleReader bundle(argv[1]);
        const auto asset = bundle.read_section("tokenizer.bpe");
        const PaligemmaBpeTokenizer tokenizer(
            parse_paligemma_bpe_asset(std::string_view(asset.data(), asset.size())));
        const auto runtime_bytes = bundle.read_section("runtime.json");
        const auto runtime = nlohmann::json::parse(runtime_bytes.begin(), runtime_bytes.end());
        PromptTokens tokens;
        tokens.image_token_id = runtime.at("image_token_id");
        tokens.image_tokens_per_image = runtime.at("num_image_tokens");
        tokens.bos_token_id = runtime.at("bos_token_id");
        tokens.reserved_text =
            reserved_text(tokenizer, runtime.at("reserved_token_ids").get<std::vector<int32_t>>(),
                          tokens.image_token_id);
        std::unordered_set<int32_t> skipped;
        for (const int32_t id : runtime.at("special_token_ids").get<std::vector<int32_t>>())
            skipped.insert(id);

        std::ifstream in(argv[2]);
        const auto cases = nlohmann::json::parse(in);
        nlohmann::json mismatches = nlohmann::json::array();
        std::size_t prompts = 0, decodes = 0;
        for (const auto& item : cases.at("prompts")) {
            PromptParts prompt;
            prompt.images.resize(item.at("images").get<std::size_t>());
            prompt.prefix = item.at("prefix").get<std::string>();
            const auto expected = item.at("input_ids").get<std::vector<int32_t>>();
            const auto actual =
                prompt_ids(prompt, tokenizer, tokens, std::numeric_limits<std::size_t>::max());
            if (actual != expected)
                mismatches.push_back(
                    {{"prefix", prompt.prefix}, {"expected", expected}, {"actual", actual}});
            ++prompts;
        }
        for (const auto& item : cases.at("decodes")) {
            const auto ids = item.at("token_ids").get<std::vector<int32_t>>();
            const auto expected = item.at("text").get<std::string>();
            const auto actual = tokenizer.decode(ids, skipped);
            if (actual != expected)
                mismatches.push_back(
                    {{"token_ids", ids}, {"expected", expected}, {"actual", actual}});
            ++decodes;
        }
        std::cout << nlohmann::json{{"prompts", prompts},
                                    {"decodes", decodes},
                                    {"mismatches", mismatches}}
                         .dump()
                  << '\n';
        return mismatches.empty() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
