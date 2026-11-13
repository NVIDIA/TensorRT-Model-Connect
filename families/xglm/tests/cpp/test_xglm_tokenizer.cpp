/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/xglm/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>

int main() {
    auto config = nlohmann::json{
        {"model",
         {{"type", "Unigram"},
          {"unk_id", 1},
          {"vocab", nlohmann::json::array(
                        {{"<s>", 0.0}, {"<unk>", -20.0}, {"</s>", 0.0}, {"▁hello", -1.0}})}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single",
           {{{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}},
            {{"Sequence", {{"id", "A"}, {"type_id", 0}}}}}}}}};
    auto source = config.dump();
    auto tokenizer = trtmc::CreateUnigramTokenizer(source.data(), source.size(), true);
    if (tokenizer->encode("hello") != std::vector<int32_t>{2, 3}) {
        std::cerr << "FAIL: a prefix-only post-processor must not append EOS\n";
        return 1;
    }
    config["post_processor"]["single"] = {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                                          {{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}}};
    config["model"]["vocab"][0][0] = "<pad>";
    source = config.dump();
    tokenizer = trtmc::CreateUnigramTokenizer(source.data(), source.size(), true);
    if (tokenizer->encode("hello") != std::vector<int32_t>{3, 2}) {
        std::cerr << "FAIL: a suffix EOS post-processor must append EOS\n";
        return 1;
    }
    return 0;
}
