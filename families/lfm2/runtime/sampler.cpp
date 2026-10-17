/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/sampler.h"

#include "trtmc/task.h"

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

float sanitize_probability(float value, float default_value) {
    if (!std::isfinite(value))
        return default_value;
    return std::min(std::max(value, 0.0F), 1.0F);
}

bool top_p_enabled(float value) {
    return value > 0.0F && value < 1.0F - kSamplingEpsilon;
}

bool distribution_sampling_enabled(const Lfm2SamplingParams& params) {
    return sanitize_temperature(params.temperature) >= kSamplingEpsilon &&
           (params.min_p > 0.0F || top_p_enabled(sanitize_probability(params.top_p, 1.0F)) ||
            params.top_k <= 0 || params.top_k > 1 || params.seed >= 0);
}

std::vector<int32_t> make_candidate_indices(const std::vector<float>& logits,
                                            int32_t candidate_count) {
    std::vector<int32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto descending = [&](int32_t lhs, int32_t rhs) {
        const float left = logits[static_cast<std::size_t>(lhs)];
        const float right = logits[static_cast<std::size_t>(rhs)];
        return left > right || (left == right && lhs < rhs);
    };
    std::partial_sort(indices.begin(), indices.begin() + candidate_count, indices.end(),
                      descending);
    // HF masks scores strictly below the kth score, retaining every boundary
    // tie. Truncating to exactly k changes the distribution for tied logits.
    const float threshold = logits[static_cast<std::size_t>(indices[candidate_count - 1])];
    indices.erase(std::remove_if(indices.begin() + candidate_count, indices.end(),
                                 [&](int32_t index) { return logits[index] < threshold; }),
                  indices.end());
    std::sort(indices.begin(), indices.end(), descending);
    return indices;
}

std::vector<float> make_candidate_probabilities(const std::vector<float>& logits,
                                                const std::vector<int32_t>& indices) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t index : indices)
        maximum = std::max(maximum, logits[static_cast<std::size_t>(index)]);
    std::vector<float> probabilities(indices.size());
    float total = 0.0F;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const float scaled = logits[static_cast<std::size_t>(indices[index])] - maximum;
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

int32_t apply_min_p_filter(const std::vector<float>& logits, const std::vector<int32_t>& indices,
                           int32_t candidate_count, float min_p) {
    if (min_p <= 0.0F)
        return candidate_count;

    // softmax(score) >= min_p * softmax(max_score) is equivalent to this
    // log-space test, without rounding a normalization twice at the boundary.
    const float threshold = logits[indices.front()] + std::log(min_p);
    int32_t keep = 0;
    while (keep < candidate_count && logits[indices[static_cast<std::size_t>(keep)]] >= threshold) {
        ++keep;
    }
    return std::max(keep, 1);
}

void apply_top_p_filter(const std::vector<float>& logits, std::vector<int32_t>& indices,
                        float top_p) {
    if (!top_p_enabled(top_p))
        return;
    // HF removes the ascending low-probability tail, including tokens exactly
    // at 1-top_p. The complementary descending-prefix formulation changes
    // which tied tokens survive at the nucleus boundary.
    std::sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
        return logits[lhs] < logits[rhs] || (logits[lhs] == logits[rhs] && lhs < rhs);
    });
    const auto probabilities = make_candidate_probabilities(logits, indices);
    float cumulative = 0.0F;
    std::size_t remove = 0;
    while (remove + 1 < indices.size()) {
        cumulative += probabilities[remove];
        if (cumulative > 1.0F - top_p)
            break;
        ++remove;
    }
    indices.erase(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(remove));
    std::sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
        return logits[lhs] > logits[rhs] || (logits[lhs] == logits[rhs] && lhs < rhs);
    });
}

Lfm2SampleResult argmax(const std::vector<float>& logits, const Lfm2SamplingParams& params) {
    Lfm2SampleResult result;
    if (logits.empty()) {
        result.is_eos = lfm2_is_eos_token(params, 0);
        return result;
    }
    const auto found = std::max_element(logits.begin(), logits.end());
    result.token_id = static_cast<int32_t>(std::distance(logits.begin(), found));
    result.logprob = *found;
    result.is_eos = lfm2_is_eos_token(params, result.token_id);
    return result;
}

Lfm2SamplingDistribution make_distribution(std::vector<float> logits,
                                           const Lfm2SamplingParams& params) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    const float temperature = sanitize_temperature(params.temperature);
    const float top_p = sanitize_probability(params.top_p, 1.0F);
    const float min_p = sanitize_probability(params.min_p, 0.0F);
    const int32_t resolved_top_k = lfm2_resolve_top_k(params);
    const int32_t candidate_count =
        resolved_top_k <= 0 ? vocab_size : std::min(resolved_top_k, vocab_size);

    // HF applies temperature to FP32 scores before top-k and softmax.
    for (float& score : logits)
        score /= temperature;
    Lfm2SamplingDistribution distribution;
    distribution.indices = make_candidate_indices(logits, candidate_count);
    // Match HF's processor order. Applying min-p first can change the nucleus
    // because it renormalizes the probabilities used by top-p.
    apply_top_p_filter(logits, distribution.indices, top_p);
    const int32_t keep = apply_min_p_filter(
        logits, distribution.indices, static_cast<int32_t>(distribution.indices.size()), min_p);
    distribution.indices.resize(static_cast<std::size_t>(keep));
    distribution.probabilities = make_candidate_probabilities(logits, distribution.indices);
    return distribution;
}

class Lfm2GreedySampler final : public Lfm2ISampler {
  public:
    Lfm2SampleResult sample(const float* logits, int32_t vocab_size,
                            const Lfm2SamplingParams& params,
                            const std::vector<int32_t>& token_history) override {
        return argmax(lfm2_apply_repetition_penalty(logits, vocab_size, params.repetition_penalty,
                                                    token_history),
                      params);
    }
};

class Lfm2DistributionSampler final : public Lfm2ISampler {
  public:
    explicit Lfm2DistributionSampler(std::uint64_t seed)
        : initial_state_(seed == 0 ? 1 : seed), state_(initial_state_) {}

    Lfm2SampleResult sample(const float* logits, int32_t vocab_size,
                            const Lfm2SamplingParams& params,
                            const std::vector<int32_t>& token_history) override {
        auto adjusted = lfm2_apply_repetition_penalty(logits, vocab_size, params.repetition_penalty,
                                                      token_history);
        if (adjusted.empty())
            return argmax(adjusted, params);
        if (sanitize_temperature(params.temperature) < kSamplingEpsilon)
            return argmax(adjusted, params);

        const auto distribution = make_distribution(std::move(adjusted), params);
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        const float sample = static_cast<float>(state_ & 0xFFFFFFFFULL) / 4294967296.0F;

        return lfm2_sample_from_distribution(distribution,
                                             std::min(sample, std::nextafter(1.0F, 0.0F)), params);
    }
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

Lfm2SamplingDistribution lfm2_sampling_distribution(const float* logits, int32_t vocab_size,
                                                    const Lfm2SamplingParams& params,
                                                    const std::vector<int32_t>& token_history) {
    auto adjusted =
        lfm2_apply_repetition_penalty(logits, vocab_size, params.repetition_penalty, token_history);
    if (adjusted.empty())
        return {};
    if (sanitize_temperature(params.temperature) < kSamplingEpsilon)
        return {{argmax(adjusted, params).token_id}, {1.0F}};
    return make_distribution(std::move(adjusted), params);
}

Lfm2SampleResult lfm2_sample_from_distribution(const Lfm2SamplingDistribution& distribution,
                                               float uniform, const Lfm2SamplingParams& params) {
    if (distribution.indices.empty() ||
        distribution.indices.size() != distribution.probabilities.size())
        throw std::invalid_argument("LFM2 sampling requires a nonempty distribution");
    if (!std::isfinite(uniform) || uniform < 0.0F || uniform >= 1.0F)
        throw std::invalid_argument("LFM2 sampling uniform must be in [0, 1)");
    float cumulative = 0.0F;
    std::size_t selected = distribution.indices.size() - 1;
    for (std::size_t index = 0; index < distribution.indices.size(); ++index) {
        cumulative += distribution.probabilities[index];
        if (uniform < cumulative) {
            selected = index;
            break;
        }
    }
    Lfm2SampleResult result;
    result.token_id = distribution.indices[selected];
    result.logprob =
        std::log(std::max(distribution.probabilities[selected], std::numeric_limits<float>::min()));
    result.is_eos = lfm2_is_eos_token(params, result.token_id);
    return result;
}

bool lfm2_is_eos_token(const Lfm2SamplingParams& params, int32_t token_id) {
    return std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token_id) !=
           params.eos_token_ids.end();
}

std::vector<float> lfm2_apply_repetition_penalty(const float* logits, int32_t vocab_size,
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

int32_t lfm2_resolve_top_k(const Lfm2SamplingParams& params) {
    // The request is authoritative. Model-card invocations that want LFM2's
    // recommended sampling policy pass top_k=50 explicitly.
    return params.top_k;
}

Lfm2SamplingParams
lfm2_sampling_params_from_config(const TextGenerationConfig& cfg,
                                 const std::vector<int32_t>& default_eos_token_ids) {
    Lfm2SamplingParams params;
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

std::unique_ptr<Lfm2ISampler> create_lfm2_sampler(const Lfm2SamplingParams& params) {
    if (!distribution_sampling_enabled(params))
        return std::make_unique<Lfm2GreedySampler>();
    const auto seed = params.seed >= 0 ? static_cast<std::uint64_t>(params.seed) : 42ULL;
    return std::make_unique<Lfm2DistributionSampler>(seed);
}

bool lfm2_sampling_uses_distribution(const Lfm2SamplingParams& params) {
    return distribution_sampling_enabled(params);
}

} // namespace trtmc
