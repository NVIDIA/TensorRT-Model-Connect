/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc::diffusion::flux_text {

struct PaddedTextInputs {
    std::vector<int32_t> input_ids;
    std::vector<float> attention_mask;
    std::size_t valid_tokens{0};
};

inline int32_t resolve_pad_token_id(ITokenizer* tokenizer, bool is_flux2) {
    if (!is_flux2 || tokenizer == nullptr) {
        return 0;
    }
    try {
        const int32_t pad_id = tokenizer->id_for_token("<pad>");
        return pad_id >= 0 ? pad_id : 0;
    } catch (const std::exception&) {
        return 0;
    }
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
