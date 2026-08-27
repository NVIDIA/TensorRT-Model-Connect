/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/sampler.h"

#include "trtmc/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace trtmc {
namespace {

constexpr float kSamplingEpsilon = 1.0e-6F;

float sanitize_temperature(float value) {
    return std::isfinite(value) ? std::max(value, 0.0F) : 1.0F;
}

float sanitize_probability(float value, float fallback) {
    if (!std::isfinite(value))
        return fallback;
    return std::min(std::max(value, 0.0F), 1.0F);
}

bool top_p_enabled(float value) {
    return value > 0.0F && value < 1.0F - kSamplingEpsilon;
}

bool distribution_sampling_enabled(const Lfm2MoeSamplingParams& params) {
    return sanitize_temperature(params.temperature) >= kSamplingEpsilon &&
           (params.min_p > 0.0F || top_p_enabled(sanitize_probability(params.top_p, 1.0F)) ||
            params.top_k <= 0 || params.top_k > 1 || params.seed >= 0);
}

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probabilities;
    int32_t keep{0};
};

std::vector<int32_t> make_candidate_indices(const std::vector<float>& logits,
                                            int32_t candidate_count) {
    std::vector<int32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + candidate_count, indices.end(),
                      [&](int32_t lhs, int32_t rhs) {
                          return logits[static_cast<std::size_t>(lhs)] >
                                 logits[static_cast<std::size_t>(rhs)];
                      });
    indices.resize(static_cast<std::size_t>(candidate_count));
    return indices;
}

std::vector<float> make_candidate_probabilities(const std::vector<float>& logits,
                                                const std::vector<int32_t>& indices,
                                                float temperature) {
    const float maximum = logits[static_cast<std::size_t>(indices.front())];
    std::vector<float> probabilities(indices.size());
    float total = 0.0F;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const float scaled =
            (logits[static_cast<std::size_t>(indices[index])] - maximum) / temperature;
        const float probability = std::isfinite(scaled) ? std::exp(scaled) : 0.0F;
        probabilities[index] = probability;
        total += probability;
    }

    if (total <= 0.0F) {
        const float uniform = 1.0F / static_cast<float>(indices.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
        return probabilities;
    }
    for (float& probability : probabilities)
        probability /= total;
    return probabilities;
}

void renormalize_prefix(std::vector<float>& probabilities, int32_t count) {
    float prefix_total = 0.0F;
    for (int32_t index = 0; index < count; ++index)
        prefix_total += probabilities[static_cast<std::size_t>(index)];
    if (prefix_total > 0.0F) {
        for (int32_t index = 0; index < count; ++index)
            probabilities[static_cast<std::size_t>(index)] /= prefix_total;
    }
}

int32_t apply_min_p_filter(std::vector<float>& probabilities, int32_t candidate_count,
                           float min_p) {
    if (min_p <= 0.0F)
        return candidate_count;

    const float threshold = min_p * probabilities.front();
    int32_t keep = 0;
    while (keep < candidate_count && probabilities[static_cast<std::size_t>(keep)] >= threshold) {
        ++keep;
    }
    keep = std::max(keep, 1);
    if (keep < candidate_count)
        renormalize_prefix(probabilities, keep);
    return keep;
}

int32_t apply_top_p_filter(const std::vector<float>& probabilities, int32_t keep, float top_p) {
    if (!top_p_enabled(top_p))
        return keep;

    float cumulative = 0.0F;
    int32_t nucleus = 0;
    while (nucleus < keep) {
        cumulative += probabilities[static_cast<std::size_t>(nucleus)];
        ++nucleus;
        if (cumulative >= top_p)
            break;
    }
    return std::max(nucleus, 1);
}

Lfm2MoeSampleResult argmax(const std::vector<float>& logits, const Lfm2MoeSamplingParams& params) {
    Lfm2MoeSampleResult result;
    if (logits.empty()) {
        result.is_eos = lfm2_moe_is_eos_token(params, 0);
        return result;
    }
    const auto found = std::max_element(logits.begin(), logits.end());
    result.token_id = static_cast<int32_t>(std::distance(logits.begin(), found));
    result.logprob = *found;
    result.is_eos = lfm2_moe_is_eos_token(params, result.token_id);
    return result;
}

FilteredDistribution make_distribution(const std::vector<float>& logits,
                                       const Lfm2MoeSamplingParams& params) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    const float temperature = sanitize_temperature(params.temperature);
    const float top_p = sanitize_probability(params.top_p, 1.0F);
    const float min_p = sanitize_probability(params.min_p, 0.0F);
    const int32_t resolved_top_k = lfm2_moe_resolve_top_k(params);
    const int32_t candidate_count =
        resolved_top_k <= 0 ? vocab_size : std::min(resolved_top_k, vocab_size);

    FilteredDistribution distribution;
    distribution.indices = make_candidate_indices(logits, candidate_count);
    distribution.probabilities =
        make_candidate_probabilities(logits, distribution.indices, temperature);
    distribution.keep = apply_min_p_filter(distribution.probabilities, candidate_count, min_p);
    distribution.keep = apply_top_p_filter(distribution.probabilities, distribution.keep, top_p);
    renormalize_prefix(distribution.probabilities, distribution.keep);
    return distribution;
}

class Lfm2MoeGreedySampler final : public Lfm2MoeISampler {
  public:
    Lfm2MoeSampleResult sample(const float* logits, int32_t vocab_size,
                               const Lfm2MoeSamplingParams& params,
                               const std::vector<int32_t>& token_history) override {
        return argmax(lfm2_moe_apply_repetition_penalty(logits, vocab_size,
                                                        params.repetition_penalty, token_history),
                      params);
    }
    const char* sampler_type() const override { return "lfm2_moe_greedy"; }
};

class Lfm2MoeDistributionSampler final : public Lfm2MoeISampler {
  public:
    explicit Lfm2MoeDistributionSampler(std::uint64_t seed)
        : initial_state_(seed == 0 ? 1 : seed), state_(initial_state_) {}

    Lfm2MoeSampleResult sample(const float* logits, int32_t vocab_size,
                               const Lfm2MoeSamplingParams& params,
                               const std::vector<int32_t>& token_history) override {
        auto adjusted = lfm2_moe_apply_repetition_penalty(logits, vocab_size,
                                                          params.repetition_penalty, token_history);
        if (adjusted.empty())
            return argmax(adjusted, params);
        if (sanitize_temperature(params.temperature) < kSamplingEpsilon)
            return argmax(adjusted, params);

        const auto distribution = make_distribution(adjusted, params);
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        const float sample = static_cast<float>(state_ & 0xFFFFFFFFULL) / 4294967296.0F;

        float cumulative = 0.0F;
        int32_t selected = distribution.keep - 1;
        for (int32_t index = 0; index < distribution.keep; ++index) {
            cumulative += distribution.probabilities[static_cast<std::size_t>(index)];
            if (sample < cumulative) {
                selected = index;
                break;
            }
        }
        Lfm2MoeSampleResult result;
        result.token_id = distribution.indices[static_cast<std::size_t>(selected)];
        result.logprob =
            std::log(std::max(distribution.probabilities[static_cast<std::size_t>(selected)],
                              std::numeric_limits<float>::min()));
        result.is_eos = lfm2_moe_is_eos_token(params, result.token_id);
        return result;
    }

    const char* sampler_type() const override { return "lfm2_moe_distribution"; }
    void reset() override { state_ = initial_state_; }

  private:
    std::uint64_t initial_state_;
    std::uint64_t state_;
};

void apply_repetition_penalty_to_history(std::vector<float>& logits, int32_t vocab_size,
                                         float penalty, const std::vector<int32_t>& token_history) {
    std::unordered_set<int32_t> seen;
    seen.reserve(token_history.size());
    for (int32_t token : token_history) {
        if (token < 0 || token >= vocab_size)
            continue;
        if (!seen.insert(token).second)
            continue;
        float& score = logits[static_cast<std::size_t>(token)];
        score = score < 0.0F ? score * penalty : score / penalty;
    }
}

} // namespace

bool lfm2_moe_is_eos_token(const Lfm2MoeSamplingParams& params, int32_t token_id) {
    return std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token_id) !=
           params.eos_token_ids.end();
}

std::vector<float> lfm2_moe_apply_repetition_penalty(const float* logits, int32_t vocab_size,
                                                     float penalty,
                                                     const std::vector<int32_t>& token_history) {
    if (vocab_size <= 0 || logits == nullptr)
        return {};
    if (!std::isfinite(penalty) || penalty <= 0.0F)
        throw std::invalid_argument("LFM2 repetition_penalty must be finite and greater than zero");

    std::vector<float> adjusted(logits, logits + vocab_size);
    if (std::abs(penalty - 1.0F) < kSamplingEpsilon)
        return adjusted;

    apply_repetition_penalty_to_history(adjusted, vocab_size, penalty, token_history);
    return adjusted;
}

int32_t lfm2_moe_resolve_top_k(const Lfm2MoeSamplingParams& params) {
    // The request is authoritative. Model-card invocations that want LFM2's
    // recommended sampling policy pass top_k=50 explicitly.
    return params.top_k;
}

Lfm2MoeSamplingParams
lfm2_moe_sampling_params_from_config(const GenerateConfig& cfg,
                                     const std::vector<int32_t>& default_eos_token_ids) {
    Lfm2MoeSamplingParams params;
    params.temperature = cfg.temperature;
    params.top_k = cfg.top_k;
    params.top_p = cfg.top_p;
    params.min_p = cfg.min_p;
    params.repetition_penalty = cfg.repetition_penalty;
    params.seed = cfg.seed;
    if (cfg.eos_token_id >= 0)
        params.eos_token_ids = {cfg.eos_token_id};
    else
        params.eos_token_ids = default_eos_token_ids;
    return params;
}

std::unique_ptr<Lfm2MoeISampler> create_lfm2_moe_sampler(const Lfm2MoeSamplingParams& params) {
    if (!distribution_sampling_enabled(params))
        return std::make_unique<Lfm2MoeGreedySampler>();
    const auto seed = params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : 42ULL;
    return std::make_unique<Lfm2MoeDistributionSampler>(seed);
}

} // namespace trtmc
