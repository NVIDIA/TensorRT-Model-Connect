/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/olmo/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>

class Tokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> ids;
    std::vector<int32_t> encode(const std::string&) const override { return ids; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return 99; }
    std::string token_for_id(int32_t) const override { return "</s>"; }
    int32_t eos_token_id() const override { return 99; }
};

int main() {
    Tokenizer tokenizer;
    tokenizer.ids = {10, 42, 99};
    // The tokenizer EOS differs from the model's generation stop ID (0).
    if (trtmc::olmo_encode_causal_prompt(tokenizer, "hello") != std::vector<int32_t>{10, 42} ||
        trtmc::olmo_encode_causal_prompt(tokenizer, "hello</s>") != tokenizer.ids) {
        std::cerr << "FAIL: remove only the automatically appended prompt EOS\n";
        return 1;
    }
    tokenizer.ids = {10, 42};
    if (trtmc::olmo_encode_causal_prompt(tokenizer, "hello") != tokenizer.ids)
        return 1;
    tokenizer.ids.clear();
    if (!trtmc::olmo_encode_causal_prompt(tokenizer, "").empty())
        return 1;

    const auto source =
        nlohmann::json{
            {"model",
             {{"type", "BPE"},
              {"vocab", {{"<|endoftext|>", 99}}},
              {"merges", nlohmann::json::array()}}},
            {"added_tokens", {{{"id", 99}, {"content", "<|endoftext|>"}, {"special", true}}}},
            {"post_processor",
             {{"type", "TemplateProcessing"},
              {"single",
               {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                {{"SpecialToken", {{"id", "<|endoftext|>"}, {"type_id", 0}}}}}}}}}
            .dump();
    auto real = trtmc::CreateBpeTokenizer(source.data(), source.size(), true);
    if (real->encode("<|endoftext|>") != std::vector<int32_t>{99, 99} ||
        trtmc::olmo_encode_causal_prompt(*real, "<|endoftext|>") != std::vector<int32_t>{99}) {
        std::cerr << "FAIL: preserve the explicit EOS and remove the automatically appended EOS\n";
        return 1;
    }
    return 0;
}
