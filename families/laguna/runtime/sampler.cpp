/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace trtmc::laguna {

namespace {

constexpr float kEpsilon = 1e-6F;

float clamp_unit(float value, float fallback) {
    return std::isfinite(value) ? std::min(std::max(value, 0.0F), 1.0F) : fallback;
}

bool top_p_enabled(float top_p) {
    return top_p > 0.0F && top_p < 1.0F;
}

SampleResult argmax(const float* logits, std::int32_t vocab_size, const SamplingParams& params) {
    const auto* best = std::max_element(logits, logits + vocab_size);
    const auto token = static_cast<std::int32_t>(best - logits);
    return {token, is_eos_token(params, token)};
}

class GreedySampler final : public Sampler {
  public:
    SampleResult sample(const float* logits, std::int32_t vocab_size,
                        const SamplingParams& params) override {
        return argmax(logits, vocab_size, params);
    }
};

class RandomSampler final : public Sampler {
  public:
    explicit RandomSampler(std::uint64_t seed) : state_(seed == 0 ? 1 : seed) {}

    SampleResult sample(const float* logits, std::int32_t vocab_size,
                        const SamplingParams& params) override {
        const float temperature =
            std::isfinite(params.temperature) ? std::max(params.temperature, 0.0F) : 1.0F;
        const float top_p = clamp_unit(params.top_p, 1.0F);
        const float min_p = clamp_unit(params.min_p, 0.0F);
        if (temperature < kEpsilon || top_p <= 0.0F)
            return argmax(logits, vocab_size, params);

        const bool full_vocab = top_p_enabled(top_p) && params.top_k <= 1;
        const std::int32_t k = (params.top_k <= 0 || full_vocab)
                                   ? vocab_size
                                   : std::min(std::max(params.top_k, 1), vocab_size);
        std::vector<std::int32_t> order(static_cast<std::size_t>(vocab_size));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
        std::vector<float> probs(static_cast<std::size_t>(k));
        const float top = logits[order[0]];
        float total = 0.0F;
        for (std::int32_t i = 0; i < k; ++i) {
            const float scaled = (logits[order[static_cast<std::size_t>(i)]] - top) / temperature;
            probs[static_cast<std::size_t>(i)] = std::isfinite(scaled) ? std::exp(scaled) : 0.0F;
            total += probs[static_cast<std::size_t>(i)];
        }
        for (auto& p : probs)
            p = total > 0.0F ? p / total : 1.0F / static_cast<float>(k);

        std::int32_t keep = k;
        if (min_p > 0.0F) {
            keep = 0;
            while (keep < k && probs[static_cast<std::size_t>(keep)] >= min_p * probs[0])
                ++keep;
            keep = std::max(keep, 1);
        }
        if (top_p_enabled(top_p)) {
            float cumulative = 0.0F;
            std::int32_t kept = 0;
            while (kept < keep) {
                cumulative += probs[static_cast<std::size_t>(kept++)];
                if (cumulative >= top_p)
                    break;
            }
            keep = std::max(kept, 1);
        }
        const float kept_total = std::accumulate(probs.begin(), probs.begin() + keep, 0.0F);

        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        const float u = static_cast<float>(state_ & 0xFFFFFFFFULL) / 4294967296.0F * kept_total;
        float cumulative = 0.0F;
        for (std::int32_t i = 0; i < keep; ++i) {
            cumulative += probs[static_cast<std::size_t>(i)];
            if (u < cumulative) {
                const auto token = order[static_cast<std::size_t>(i)];
                return {token, is_eos_token(params, token)};
            }
        }
        const auto token = order[static_cast<std::size_t>(keep - 1)];
        return {token, is_eos_token(params, token)};
    }

  private:
    std::uint64_t state_;
};

} // namespace

bool is_eos_token(const SamplingParams& params, std::int32_t token_id) {
    return std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token_id) !=
           params.eos_token_ids.end();
}

std::unique_ptr<Sampler> create_sampler(const SamplingParams& params) {
    const float top_p = clamp_unit(params.top_p, 1.0F);
    const float min_p = clamp_unit(params.min_p, 0.0F);
    if (params.top_k <= 1 && !top_p_enabled(top_p) && min_p <= 0.0F && params.seed < 0)
        return std::make_unique<GreedySampler>();
    return std::make_unique<RandomSampler>(
        params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : 42ULL);
}

} // namespace trtmc::laguna
