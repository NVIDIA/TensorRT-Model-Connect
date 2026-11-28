/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc::openai_gpt {
namespace {

constexpr std::uint64_t kDefaultSeed = 42;

} // namespace

Sampler::Sampler(const SamplingParams& params)
    : params_(params), greedy_(params.temperature <= 0.0 ||
                               (params.top_k == 1 && params.top_p >= 1.0 && params.min_p <= 0.0)),
      state_(params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : kDefaultSeed) {
    if (!std::isfinite(params.temperature) || params.temperature < 0.0 || params.top_k < 0 ||
        !(params.top_p >= 0.0 && params.top_p <= 1.0) ||
        !(params.min_p >= 0.0 && params.min_p <= 1.0))
        throw std::invalid_argument("OpenAI GPT sampling parameters are out of range");
}

double Sampler::uniform() {
    // SplitMix64: every seed, small ones included, gives a well-mixed stream.
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t mixed = state_;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    mixed ^= mixed >> 31;
    // The top 53 bits, exactly representable: a value in [0, 1).
    return static_cast<double>(mixed >> 11) * 0x1.0p-53;
}

std::int32_t Sampler::sample(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::invalid_argument("OpenAI GPT sampler received no logits");
    if (!std::all_of(logits.begin(), logits.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("OpenAI GPT engine produced a NaN or infinite logit");
    if (greedy_)
        return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                         logits.begin());
    const auto vocab = static_cast<std::int32_t>(logits.size());
    const std::int32_t k = params_.top_k == 0 ? vocab : std::min(params_.top_k, vocab);
    std::vector<std::int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](std::int32_t a, std::int32_t b) {
                          const auto la = logits[static_cast<std::size_t>(a)];
                          const auto lb = logits[static_cast<std::size_t>(b)];
                          return la > lb || (la == lb && a < b);
                      });
    // Probabilities relative to the most likely token; the largest is exactly 1.
    std::vector<double> weights(static_cast<std::size_t>(k));
    const double top = logits[static_cast<std::size_t>(order[0])];
    for (std::int32_t index = 0; index < k; ++index) {
        const double logit =
            logits[static_cast<std::size_t>(order[static_cast<std::size_t>(index)])];
        weights[static_cast<std::size_t>(index)] = std::exp((logit - top) / params_.temperature);
    }
    std::int32_t keep = k;
    if (params_.min_p > 0.0) {
        keep = 1;
        while (keep < k && weights[static_cast<std::size_t>(keep)] >= params_.min_p)
            ++keep;
    }
    if (params_.top_p < 1.0) {
        const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        double cumulative = 0.0;
        std::int32_t nucleus = 0;
        while (nucleus < keep) {
            cumulative += weights[static_cast<std::size_t>(nucleus++)];
            if (cumulative >= params_.top_p * total)
                break;
        }
        keep = nucleus;
    }
    const double kept = std::accumulate(weights.begin(), weights.begin() + keep, 0.0);
    const double target = uniform() * kept;
    double cumulative = 0.0;
    for (std::int32_t index = 0; index + 1 < keep; ++index) {
        cumulative += weights[static_cast<std::size_t>(index)];
        if (target < cumulative)
            return order[static_cast<std::size_t>(index)];
    }
    return order[static_cast<std::size_t>(keep - 1)];
}

} // namespace trtmc::openai_gpt
