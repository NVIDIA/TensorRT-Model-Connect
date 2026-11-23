/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/flux/runtime/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::diffusion::flux_text {

struct PaddedTextInputs {
    std::vector<int32_t> input_ids;
    std::vector<float> attention_mask;
    std::size_t valid_tokens{0};
};

inline int32_t resolve_pad_token_id(ITokenizer* tokenizer, bool is_flux2,
                                    const std::string& pad_token = "<pad>") {
    if (!is_flux2) {
        return 0;
    }
    if (tokenizer == nullptr)
        throw std::runtime_error("FLUX.2 tokenizer is missing");
    const int32_t pad_id = tokenizer->id_for_token(pad_token);
    if (pad_id < 0)
        throw std::runtime_error("FLUX.2 tokenizer has no " + pad_token + " token");
    return pad_id;
}

// FLUX.2 prompt templates. Mistral 3 (FLUX.2-dev) uses its system prompt;
// Qwen3 (FLUX.2-klein) applies the chat template with thinking disabled.
inline std::string apply_flux2_template(const std::string& prompt,
                                        const std::string& text_template) {
    if (text_template == "qwen3_chat") {
        return "<|im_start|>user\n" + prompt +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    if (!text_template.empty())
        throw std::runtime_error("unknown FLUX.2 text_template: " + text_template);
    static const char* kSystemMsg = "You are an AI that reasons about image descriptions. "
                                    "You give structured responses focusing on object "
                                    "relationships, object\n"
                                    "attribution and actions without speculation.";
    return std::string("<s>[SYSTEM_PROMPT]") + kSystemMsg + "[/SYSTEM_PROMPT][INST]" + prompt +
           "[/INST]";
}

// True classifier-free guidance: out = neg + scale * (pos - neg), in place on pos.
inline void combine_true_cfg(std::vector<float>& positive, const std::vector<float>& negative,
                             float scale) {
    if (positive.size() != negative.size())
        throw std::runtime_error("FLUX.2 CFG branch outputs differ in size");
    for (std::size_t i = 0; i < positive.size(); ++i)
        positive[i] = negative[i] + scale * (positive[i] - negative[i]);
}

inline PaddedTextInputs prepare_inputs(const std::vector<int32_t>& input_ids, int32_t seq_len,
                                       int32_t pad_token_id) {
    if (seq_len <= 0) {
        throw std::invalid_argument("FLUX text sequence length must be positive");
    }
    PaddedTextInputs prepared;
    prepared.input_ids.assign(static_cast<std::size_t>(seq_len), pad_token_id);
    prepared.attention_mask.assign(static_cast<std::size_t>(seq_len), -1e9F);
    prepared.valid_tokens = std::min(static_cast<std::size_t>(seq_len), input_ids.size());
    std::copy_n(input_ids.begin(), prepared.valid_tokens, prepared.input_ids.begin());
    std::fill_n(prepared.attention_mask.begin(), prepared.valid_tokens, 0.0F);
    return prepared;
}

inline void clear_padding_rows(std::vector<float>& embeddings,
                               const std::vector<std::size_t>& valid_tokens, int32_t seq_len,
                               int32_t embedding_dim, bool preserve_padding_rows) {
    if (preserve_padding_rows) {
        return;
    }
    const auto seq = static_cast<std::size_t>(seq_len);
    const auto dim = static_cast<std::size_t>(embedding_dim);
    for (std::size_t batch = 0; batch < valid_tokens.size(); ++batch) {
        const auto first_padding = std::min(valid_tokens[batch], seq);
        auto* first = embeddings.data() + (batch * seq + first_padding) * dim;
        std::fill(first, embeddings.data() + (batch + 1) * seq * dim, 0.0F);
    }
}

} // namespace trtmc::diffusion::flux_text
