/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/routing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace trtmc::qwen3_next {

std::vector<std::int32_t> select_experts(const float* probabilities, std::int32_t num_experts,
                                         std::int32_t top_k) {
    if (num_experts < 0 || top_k < 0 || top_k > num_experts ||
        (num_experts != 0 && probabilities == nullptr))
        throw std::invalid_argument("invalid Qwen3-Next router dimensions");
    if (top_k == 0)
        return {};

    using Entry = std::pair<float, std::int64_t>;
    std::vector<Entry> entries(static_cast<std::size_t>(num_experts));
    for (std::int32_t i = 0; i < num_experts; ++i)
        entries[i] = {probabilities[i], i};
    const auto greater = [](const Entry& a, const Entry& b) {
        return (std::isnan(a.first) && !std::isnan(b.first)) || a.first > b.first;
    };
    // ATen's CPU TopK uses this threshold and a score-only comparator. Adding
    // an index tie-break would select different experts for equal BF16 logits.
    if (static_cast<std::int64_t>(top_k) * 64 <= num_experts) {
        std::partial_sort(entries.begin(), entries.begin() + top_k, entries.end(), greater);
    } else {
        std::nth_element(entries.begin(), entries.begin() + top_k - 1, entries.end(), greater);
        std::sort(entries.begin(), entries.begin() + top_k - 1, greater);
    }

    std::vector<std::int32_t> selected(static_cast<std::size_t>(top_k));
    for (std::int32_t i = 0; i < top_k; ++i)
        selected[i] = static_cast<std::int32_t>(entries[i].second);
    return selected;
}

} // namespace trtmc::qwen3_next
