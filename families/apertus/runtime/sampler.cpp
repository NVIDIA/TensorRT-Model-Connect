/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/apertus/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc::apertus {
namespace {

constexpr float kEpsilon = 1e-6F;
constexpr std::uint64_t kDefaultSeed = 42;

std::int32_t argmax(const std::vector<float>& logits) {
    return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                     logits.begin());
}

} // namespace

Sampler::Sampler(const SamplingParams& params)
    : params_(params),
      greedy_(params.temperature < kEpsilon || params.top_p <= 0.0F ||
              (params.top_k == 1 && params.top_p >= 1.0F - kEpsilon && params.min_p <= 0.0F)),
      state_(params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : kDefaultSeed) {
    if (state_ == 0)
        state_ = 1;
}

float Sampler::uniform() {
    // xorshift64: deterministic for a given seed.
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return static_cast<float>(state_ & 0xFFFFFFFFULL) / 4294967296.0F;
}

std::int32_t Sampler::sample(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::invalid_argument("Apertus sampler received no logits");
    if (greedy_)
        return argmax(logits);
    const auto vocab = static_cast<std::int32_t>(logits.size());
    const std::int32_t k = params_.top_k <= 0 ? vocab : std::min(params_.top_k, vocab);
    std::vector<std::int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](std::int32_t a, std::int32_t b) {
                          const auto la = logits[static_cast<std::size_t>(a)];
                          const auto lb = logits[static_cast<std::size_t>(b)];
                          return la > lb || (la == lb && a < b);
                      });
    std::vector<float> probabilities(static_cast<std::size_t>(k));
    const float top = logits[static_cast<std::size_t>(order[0])];
    float total = 0.0F;
    for (std::int32_t index = 0; index < k; ++index) {
        const auto value = std::exp(
            (logits[static_cast<std::size_t>(order[static_cast<std::size_t>(index)])] - top) /
            params_.temperature);
        probabilities[static_cast<std::size_t>(index)] = std::isfinite(value) ? value : 0.0F;
        total += probabilities[static_cast<std::size_t>(index)];
    }
    for (auto& probability : probabilities)
        probability /= total;
    std::int32_t keep = k;
    if (params_.min_p > 0.0F) {
        const float floor = params_.min_p * probabilities[0];
        keep = 1;
        while (keep < k && probabilities[static_cast<std::size_t>(keep)] >= floor)
            ++keep;
    }
    if (params_.top_p < 1.0F - kEpsilon) {
        float cumulative = 0.0F;
        std::int32_t nucleus = 0;
        while (nucleus < keep) {
            cumulative += probabilities[static_cast<std::size_t>(nucleus++)];
            if (cumulative >= params_.top_p)
                break;
        }
        keep = std::max(nucleus, 1);
    }
    const float kept = std::accumulate(probabilities.begin(), probabilities.begin() + keep, 0.0F);
    const float target = uniform() * kept;
    float cumulative = 0.0F;
    for (std::int32_t index = 0; index < keep; ++index) {
        cumulative += probabilities[static_cast<std::size_t>(index)];
        if (target < cumulative)
            return order[static_cast<std::size_t>(index)];
    }
    return order[static_cast<std::size_t>(keep - 1)];
}

} // namespace trtmc::apertus
