/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/diffusion_gemma/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const std::vector<std::string> vocabulary = {
            "<0x41>", "<0xE2>",        "<0x82>", "<0xAC>", "<0xFF>", "<",      "0x41",
            ">",      "literal<0x41>", "<0xF0>", "<0x9F>", "<0xC0>", "<0xAF>", "<0xED>",
            "<0xA0>", "<0x80>",        "▁word",  "<0x96>", "<0x81>",
        };
        auto vocab = nlohmann::json::object();
        for (std::size_t i = 0; i < vocabulary.size(); ++i)
            vocab[vocabulary[i]] = i;
        const nlohmann::json document{
            {"model",
             {{"type", "BPE"},
              {"vocab", vocab},
              {"merges", nlohmann::json::array()},
              {"byte_fallback", true}}},
            {"decoder",
             {{"type", "Sequence"},
              {"decoders",
               nlohmann::json::array({
                   {{"type", "Replace"}, {"pattern", {{"String", "▁"}}}, {"content", " "}},
                   {{"type", "ByteFallback"}},
                   {{"type", "Fuse"}},
                   {{"type", "Strip"}, {"content", " "}, {"start", 1}, {"stop", 0}},
               })}}},
        };
        const auto text = document.dump();
        const auto tokenizer = trtmc::diffusion_gemma::CreateBpeTokenizer(text.data(), text.size());
        const std::string replacement = "\xef\xbf\xbd";
        const std::vector<std::pair<std::vector<std::int32_t>, std::string>> cases = {
            {{0}, "A"},
            {{1, 2, 3}, "€"},
            {{1}, replacement},
            {{9, 10}, replacement + replacement},
            {{0, 4}, replacement + replacement},
            {{11, 12}, replacement + replacement},
            {{13, 14, 15}, replacement + replacement + replacement},
            {{5, 6, 7}, "<0x41>"},
            {{8}, "literal<0x41>"},
            {{16}, "word"},
            {{1, 17, 18}, "▁"},
            {{1, 8}, replacement + "literal<0x41>"},
        };
        for (const auto& [ids, expected] : cases) {
            const auto decoded = tokenizer->decode(ids);
            if (decoded != expected)
                throw std::runtime_error("ByteFallback token/run semantics changed");
            (void)nlohmann::json{{"text", decoded}}.dump();
        }
        std::cout << "diffusion_gemma ByteFallback contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
