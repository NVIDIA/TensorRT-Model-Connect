/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minicpmv4_7/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace trtmc::minicpmv4_7 {

Sampler::Sampler(const SamplingConfig& config) : config_(config) {
    if (!(config.temperature >= 0.0) || !std::isfinite(config.temperature) || config.top_k < 0 ||
        !(config.top_p > 0.0 && config.top_p <= 1.0))
        throw std::invalid_argument(
            "MiniCPM-V 4.7 sampling needs temperature >= 0, top_k >= 0 and 0 < top_p <= 1");
    generator_.seed(config.seed >= 0 ? static_cast<uint64_t>(config.seed) : std::random_device{}());
}

int32_t Sampler::sample(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::invalid_argument("MiniCPM-V 4.7 sampler received no logits");
    const auto greedy = [&] {
        return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                    logits.begin());
    };
    if (config_.top_k == 1 || config_.temperature == 0.0)
        return greedy();
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::size_t keep = order.size();
    if (config_.top_k > 0)
        keep = std::min<std::size_t>(keep, static_cast<std::size_t>(config_.top_k));
    // Descending logit, equal logits by id; only the kept candidates need ordering.
    const auto before = [&](int32_t a, int32_t b) {
        const float left = logits[static_cast<std::size_t>(a)];
        const float right = logits[static_cast<std::size_t>(b)];
        return left > right || (left == right && a < b);
    };
    if (keep < order.size())
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep),
                          order.end(), before);
    else
        std::sort(order.begin(), order.end(), before);
    const double top = logits[static_cast<std::size_t>(order.front())];
    std::vector<double> weights(keep);
    double total = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        weights[i] =
            std::exp((logits[static_cast<std::size_t>(order[i])] - top) / config_.temperature);
        total += weights[i];
    }
    double cumulative = 0.0;
    std::size_t nucleus = keep;
    for (std::size_t i = 0; i < keep; ++i) {
        cumulative += weights[i] / total;
        if (cumulative >= config_.top_p) {
            nucleus = i + 1;
            break;
        }
    }
    std::discrete_distribution<std::size_t> draw(weights.begin(), weights.begin() + nucleus);
    return order[draw(generator_)];
}

} // namespace trtmc::minicpmv4_7
