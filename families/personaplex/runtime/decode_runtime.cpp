/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace trtmc {

int32_t personaplex_select_argmax_token(const std::vector<float>& logits) {
    if (logits.empty()) {
        return 0;
    }
    const auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<int32_t>(std::distance(logits.begin(), it));
}

int32_t personaplex_sample_token_topk(const std::vector<float>& logits, float temperature,
                                      int32_t top_k, uint64_t& rng_state) {
    if (logits.empty())
        return 0;

    const auto n = static_cast<int32_t>(logits.size());

    // Fallback to argmax if temperature is near zero
    if (temperature < 1e-6F) {
        return personaplex_select_argmax_token(logits);
    }

    // Build index array sorted by logit value (descending)
    std::vector<int32_t> indices(static_cast<std::size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    const int32_t k = std::min(std::max(top_k, 1), n);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(), [&](int32_t a, int32_t b) {
            return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
        });

    // Temperature-scaled softmax over top-k
    float max_logit = logits[static_cast<std::size_t>(indices[0])];
    std::vector<float> probs(static_cast<std::size_t>(k));
    float sum = 0.0F;
    for (int32_t i = 0; i < k; ++i) {
        float scaled = (logits[static_cast<std::size_t>(indices[i])] - max_logit) / temperature;
        probs[i] = std::exp(scaled);
        sum += probs[i];
    }

    // Normalize
    if (sum > 0.0F) {
        for (int32_t i = 0; i < k; ++i)
            probs[i] /= sum;
    } else {
        // Degenerate case: uniform
        for (int32_t i = 0; i < k; ++i)
            probs[i] = 1.0F / static_cast<float>(k);
    }

    // xorshift64 random number generation
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    float u = static_cast<float>(rng_state & 0xFFFFFFFF) / 4294967296.0F;

    // Sample from cumulative distribution
    float cumulative = 0.0F;
    for (int32_t i = 0; i < k; ++i) {
        cumulative += probs[i];
        if (u < cumulative) {
            return indices[i];
        }
    }
    return indices[k - 1];
}

std::vector<int32_t> personaplex_select_topk_tokens(const std::vector<float>& logits, int32_t k) {
    if (logits.empty() || k <= 0) {
        return {};
    }

    const int32_t capped = std::min(k, static_cast<int32_t>(logits.size()));
    std::vector<int32_t> indices(logits.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }

    std::partial_sort(
        indices.begin(), indices.begin() + capped, indices.end(), [&](int32_t a, int32_t b) {
            return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
        });
    indices.resize(static_cast<std::size_t>(capped));
    return indices;
}

std::vector<float> personaplex_build_attention_mask(int32_t cache_length, int32_t max_cache_length,
                                                    bool include_current_slot) {
    const int32_t width = max_cache_length + (include_current_slot ? 1 : 0);
    if (width <= 0) {
        return {};
    }

    std::vector<float> mask(static_cast<std::size_t>(width), PersonaplexMaskedScore);
    const int32_t valid = std::max(0, std::min(cache_length, max_cache_length));
    for (int32_t i = 0; i < valid; ++i) {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }

    if (include_current_slot) {
        mask.back() = 0.0F;
    } else if (valid <= 0) {
        mask[0] = 0.0F;
    }

    return mask;
}

} // namespace trtmc
