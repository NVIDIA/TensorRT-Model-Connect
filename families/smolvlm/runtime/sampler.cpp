/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace trtmc::smolvlm {

namespace {
constexpr uint64_t kDefaultSeed = 42;
} // namespace

Sampler::Sampler(const SamplingParams& params)
    : params_(params), state_(params.seed < 0 ? kDefaultSeed : static_cast<uint64_t>(params.seed)) {
    if (!std::isfinite(params.temperature) || params.temperature < 0 || params.top_k < 0 ||
        !std::isfinite(params.top_p) || params.top_p <= 0 || params.top_p > 1 || params.seed < -1)
        throw std::invalid_argument(
            "SmolVLM sampling needs temperature >= 0, top_k >= 0, 0 < top_p <= 1, seed >= -1");
    if (state_ == 0)
        state_ = 1; // xorshift64 has no zero state.
}

bool Sampler::greedy() const {
    return params_.temperature == 0.0 || params_.top_k == 1;
}

int32_t Sampler::next(const float* logits, int32_t vocab_size) {
    if (logits == nullptr || vocab_size <= 0)
        throw std::invalid_argument("SmolVLM sampler needs logits");
    if (greedy())
        return static_cast<int32_t>(std::max_element(logits, logits + vocab_size) - logits);
    const int32_t keep = params_.top_k == 0
                             ? vocab_size
                             : static_cast<int32_t>(std::min<int64_t>(params_.top_k, vocab_size));
    std::vector<int32_t> order(static_cast<size_t>(vocab_size));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    std::vector<double> probability(static_cast<size_t>(keep));
    const double top = logits[order[0]];
    double total = 0;
    for (int32_t i = 0; i < keep; ++i) {
        probability[static_cast<size_t>(i)] =
            std::exp((logits[order[i]] - top) / params_.temperature);
        total += probability[static_cast<size_t>(i)];
    }
    int32_t kept = keep;
    double mass = 0;
    for (int32_t i = 0; i < keep; ++i) {
        mass += probability[static_cast<size_t>(i)] / total;
        if (mass >= params_.top_p) {
            kept = i + 1;
            break;
        }
    }
    double kept_total = 0;
    for (int32_t i = 0; i < kept; ++i)
        kept_total += probability[static_cast<size_t>(i)];
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    const double draw = static_cast<double>(state_ >> 11) / 9007199254740992.0 * kept_total;
    double cumulative = 0;
    for (int32_t i = 0; i < kept; ++i) {
        cumulative += probability[static_cast<size_t>(i)];
        if (draw < cumulative)
            return order[static_cast<size_t>(i)];
    }
    return order[static_cast<size_t>(kept - 1)];
}

} // namespace trtmc::smolvlm
