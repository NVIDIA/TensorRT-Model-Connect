/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compares the native Pi 0.5 prompt tokenizer, loaded from a built bundle, with LeRobot's own
// token ids in tests/data/tokenizer_golden.json. Using the bundle's tokenizer.bpe means the
// check needs no network and covers exactly what the runtime will load.
//
//   trtmc_lerobot_pi05_tokenizer_check BUNDLE GOLDEN.json
//
// Prints one JSON line with the case count and any mismatches; exits non-zero on a mismatch.

#include "families/lerobot_pi05/runtime/paligemma_bpe.h"
#include "trtmc/bundle.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: trtmc_lerobot_pi05_tokenizer_check BUNDLE GOLDEN.json\n";
        return 2;
    }
    try {
        using trtmc::lerobot_pi05::PaligemmaBpeTokenizer;
        const trtmc::BundleReader bundle(argv[1]);
        const auto asset = bundle.read_section("tokenizer.bpe");
        const PaligemmaBpeTokenizer tokenizer(trtmc::lerobot_pi05::parse_paligemma_bpe_asset(
            std::string_view(asset.data(), asset.size())));

        std::ifstream in(argv[2]);
        const auto golden = nlohmann::json::parse(in);
        const auto max_length = golden.at("max_length").get<std::size_t>();
        nlohmann::json mismatches = nlohmann::json::array();
        std::size_t index = 0;
        for (const auto& item : golden.at("cases")) {
            const auto text = item.at("text").get<std::string>();
            const auto state = item.at("state").get<std::vector<float>>();
            const auto expected = item.at("token_ids").get<std::vector<std::int32_t>>();
            const auto prompt = tokenizer.tokenize_pi05(text, state, max_length);
            std::vector<std::int32_t> actual;
            for (std::size_t position = 0; position < prompt.token_ids.size(); ++position)
                if (prompt.token_mask[position] != 0)
                    actual.push_back(prompt.token_ids[position]);
            const bool truncated = item.at("truncated").get<bool>();
            if (actual != expected || prompt.truncated != truncated ||
                prompt.token_ids.size() != max_length)
                mismatches.push_back({{"case", index},
                                      {"text", text},
                                      {"expected", expected},
                                      {"actual", actual},
                                      {"truncated", prompt.truncated}});
            ++index;
        }
        std::cout << nlohmann::json{{"cases", index}, {"mismatches", mismatches}}.dump() << '\n';
        return mismatches.empty() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
