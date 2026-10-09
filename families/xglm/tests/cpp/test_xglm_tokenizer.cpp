/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/xglm/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>

int main() {
    const auto source =
        nlohmann::json{
            {"model",
             {{"type", "Unigram"},
              {"unk_id", 1},
              {"vocab", nlohmann::json::array(
                            {{"<s>", 0.0}, {"<unk>", -20.0}, {"</s>", 0.0}, {"▁hello", -1.0}})}}},
            {"post_processor",
             {{"type", "TemplateProcessing"},
              {"single",
               {{{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}},
                {{"Sequence", {{"id", "A"}, {"type_id", 0}}}}}}}}}
            .dump();
    auto tokenizer = trtmc::CreateUnigramTokenizer(source.data(), source.size(), true);
    if (tokenizer->encode("hello") != std::vector<int32_t>{2, 3}) {
        std::cerr << "FAIL: a prefix-only post-processor must not append EOS\n";
        return 1;
    }
    return 0;
}
