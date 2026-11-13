/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::minicpmv4_6 {

SamplingOptions sampling_options(double temperature, std::int64_t top_k, double top_p,
                                 std::int64_t seed) {
    constexpr double kFloatMax = std::numeric_limits<float>::max();
    if (!std::isfinite(temperature) || temperature < 0.0 || temperature > kFloatMax ||
        (temperature > 0.0 && static_cast<float>(temperature) == 0.0F))
        throw std::invalid_argument("MiniCPM-V 4.6 temperature must be 0 or a positive FP32 value");
    if (!std::isfinite(top_p) || !(top_p > 0.0) || top_p > 1.0 || static_cast<float>(top_p) == 0.0F)
        throw std::invalid_argument("MiniCPM-V 4.6 top_p must be in (0, 1]");
    if (top_k < 0 || top_k > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("MiniCPM-V 4.6 top_k must be a non-negative int32");
    if (seed < -1)
        throw std::invalid_argument("MiniCPM-V 4.6 seed must be -1 or non-negative");
    SamplingOptions options;
    options.temperature = static_cast<float>(temperature);
    options.top_k = static_cast<std::int32_t>(top_k);
    options.top_p = static_cast<float>(top_p);
    options.seed = seed;
    return options;
}

Sampler::Sampler(const SamplingOptions& options) : options_(options) {
    if (!std::isfinite(options.temperature) || options.temperature < 0.0F ||
        !std::isfinite(options.top_p) || options.top_p <= 0.0F || options.top_p > 1.0F ||
        options.top_k < 0 || options.seed < -1)
        throw std::invalid_argument("MiniCPM-V 4.6 sampling requires temperature >= 0, top_k >= 0, "
                                    "top_p in (0, 1] and seed >= -1");
    // top_k == 0 keeps the whole vocabulary; top_k == 1 is greedy.
    greedy_ = options.temperature == 0.0F || options.top_k == 1;
    const auto seed = options.seed >= 0 ? static_cast<std::uint64_t>(options.seed) : 42ULL;
    state_ = seed == 0 ? 1 : seed;
}

std::int32_t Sampler::sample(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::invalid_argument("MiniCPM-V 4.6 sampler received no logits");
    // NaN breaks both the argmax and the sort's ordering; infinity breaks the probabilities.
    if (std::any_of(logits.begin(), logits.end(),
                    [](float value) { return !std::isfinite(value); }))
        throw std::runtime_error("MiniCPM-V 4.6 decoder produced nonfinite logits");
    if (greedy_)
        return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                         logits.begin());
    const auto vocab = static_cast<std::int32_t>(logits.size());
    const std::int32_t k = options_.top_k == 0 ? vocab : std::min(options_.top_k, vocab);
    std::vector<std::int32_t> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
    std::vector<double> probabilities(static_cast<std::size_t>(k));
    const double top = logits[static_cast<std::size_t>(order[0])];
    double total = 0.0;
    for (std::int32_t i = 0; i < k; ++i) {
        probabilities[i] = std::exp((logits[static_cast<std::size_t>(order[i])] - top) /
                                    static_cast<double>(options_.temperature));
        total += probabilities[i];
    }
    std::int32_t keep = k;
    if (options_.top_p < 1.0F) {
        double cumulative = 0.0;
        for (keep = 0; keep < k;) {
            cumulative += probabilities[keep++] / total;
            if (cumulative >= options_.top_p)
                break;
        }
    }
    double kept = 0.0;
    for (std::int32_t i = 0; i < keep; ++i)
        kept += probabilities[i];
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    const double draw = static_cast<double>(state_ & 0xFFFFFFFFULL) / 4294967296.0 * kept;
    double cumulative = 0.0;
    for (std::int32_t i = 0; i < keep; ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative)
            return order[i];
    }
    return order[keep - 1];
}

} // namespace trtmc::minicpmv4_6
