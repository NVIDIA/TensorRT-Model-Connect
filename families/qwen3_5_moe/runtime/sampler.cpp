/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5_moe/runtime/sampler.h"

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

Qwen35MoeSampleResult argmax_over_logits(const float* logits, int32_t vocab_size,
                                         const Qwen35MoeSamplingParams& params) {
    Qwen35MoeSampleResult result;
    const float* best = logits;
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > *best)
            best = logits + i;
    }
    result.token_id = static_cast<int32_t>(best - logits);
    result.logprob = *best;
    result.is_eos = params.is_eos(result.token_id);
    return result;
}

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probs;
    int32_t keep{0};
};

constexpr float kSamplingEpsilon = 1e-6F;

float sanitized_temperature(float temperature) {
    if (!std::isfinite(temperature))
        return 1.0F;
    return std::max(temperature, 0.0F);
}

float sanitized_top_p(float top_p) {
    if (!std::isfinite(top_p))
        return 1.0F;
    return std::min(std::max(top_p, 0.0F), 1.0F);
}

float sanitized_min_p(float min_p) {
    if (!std::isfinite(min_p))
        return 0.0F;
    return std::min(std::max(min_p, 0.0F), 1.0F);
}

bool top_p_enabled(float top_p) {
    return top_p > 0.0F && top_p < 1.0F - kSamplingEpsilon;
}

bool greedy_equivalent(const Qwen35MoeSamplingParams& params) {
    const float temperature = sanitized_temperature(params.temperature);
    const float top_p = sanitized_top_p(params.top_p);
    return temperature < kSamplingEpsilon || top_p <= 0.0F;
}

void topk_indices_and_softmax(FilteredDistribution& dist, const float* logits, int32_t n, int32_t k,
                              float temperature) {
    dist.indices.resize(static_cast<std::size_t>(n));
    std::iota(dist.indices.begin(), dist.indices.end(), 0);
    std::partial_sort(dist.indices.begin(), dist.indices.begin() + k, dist.indices.end(),
                      [&](int32_t a, int32_t b) {
                          return logits[static_cast<std::size_t>(a)] >
                                 logits[static_cast<std::size_t>(b)];
                      });
    const float max_logit = logits[static_cast<std::size_t>(dist.indices[0])];
    dist.probs.resize(static_cast<std::size_t>(k));
    float sum = 0.0F;
    for (int32_t i = 0; i < k; ++i) {
        const float scaled =
            (logits[static_cast<std::size_t>(dist.indices[static_cast<std::size_t>(i)])] -
             max_logit) /
            temperature;
        dist.probs[static_cast<std::size_t>(i)] = std::isfinite(scaled) ? std::exp(scaled) : 0.0F;
        sum += dist.probs[static_cast<std::size_t>(i)];
    }
    if (sum > 0.0F) {
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] /= sum;
    } else {
        const float uniform = 1.0F / static_cast<float>(k);
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] = uniform;
    }
}

int32_t apply_min_p(const FilteredDistribution& dist, int32_t k, float min_p) {
    if (min_p <= 0.0F)
        return k;
    const float max_prob = dist.probs.empty() ? 1.0F : dist.probs[0];
    if (max_prob <= 0.0F)
        return k;
    const float min_prob = min_p * max_prob;
    int32_t keep = 0;
    while (keep < k && dist.probs[static_cast<std::size_t>(keep)] >= min_prob)
        ++keep;
    return std::max(keep, 1);
}

int32_t apply_top_p(const FilteredDistribution& dist, int32_t keep, float top_p) {
    if (!top_p_enabled(top_p))
        return keep;
    float cumulative = 0.0F;
    int32_t top_p_keep = 0;
    while (top_p_keep < keep) {
        cumulative += dist.probs[static_cast<std::size_t>(top_p_keep)];
        ++top_p_keep;
        if (cumulative >= top_p)
            break;
    }
    return std::max(top_p_keep, 1);
}

void renormalize_kept_prefix(FilteredDistribution& dist, int32_t keep) {
    float kept_sum = 0.0F;
    for (int32_t i = 0; i < keep; ++i)
        kept_sum += dist.probs[static_cast<std::size_t>(i)];
    if (kept_sum > 0.0F) {
        for (int32_t i = 0; i < keep; ++i)
            dist.probs[static_cast<std::size_t>(i)] /= kept_sum;
    } else {
        const float uniform = 1.0F / static_cast<float>(keep);
        for (int32_t i = 0; i < keep; ++i)
            dist.probs[static_cast<std::size_t>(i)] = uniform;
    }
}

FilteredDistribution build_filtered_distribution(const float* logits, int32_t vocab_size,
                                                 const Qwen35MoeSamplingParams& params) {
    const int32_t n = vocab_size;
    const float temperature = sanitized_temperature(params.temperature);
    const float top_p = sanitized_top_p(params.top_p);
    const float min_p = sanitized_min_p(params.min_p);
    const bool full_vocab_for_top_p = top_p_enabled(top_p) && params.top_k <= 1;
    const int32_t k =
        (params.top_k <= 0 || full_vocab_for_top_p) ? n : std::min(std::max(params.top_k, 1), n);
    FilteredDistribution dist;
    topk_indices_and_softmax(dist, logits, n, k, temperature);
    int32_t keep = apply_min_p(dist, k, min_p);
    keep = apply_top_p(dist, keep, top_p);
    if (keep < k)
        renormalize_kept_prefix(dist, keep);
    dist.keep = keep;
    return dist;
}

class Qwen35MoeGreedySampler final : public Qwen35MoeISampler {
  public:
    Qwen35MoeSampleResult sample(const float* logits, int32_t vocab_size,
                                 const Qwen35MoeSamplingParams& params) override {
        if (vocab_size <= 0 || logits == nullptr) {
            Qwen35MoeSampleResult result;
            result.token_id = 0;
            result.is_eos = params.is_eos(0);
            return result;
        }
        return argmax_over_logits(logits, vocab_size, params);
    }
};

class Qwen35MoeTopKSampler final : public Qwen35MoeISampler {
  public:
    explicit Qwen35MoeTopKSampler(uint64_t initial_seed)
        : rng_state_(initial_seed == 0 ? 1 : initial_seed),
          initial_seed_(initial_seed == 0 ? 1 : initial_seed) {}

    Qwen35MoeSampleResult sample(const float* logits, int32_t vocab_size,
                                 const Qwen35MoeSamplingParams& params) override {
        Qwen35MoeSampleResult result;
        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = params.is_eos(0);
            return result;
        }

        if (greedy_equivalent(params))
            return argmax_over_logits(logits, vocab_size, params);

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);

        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        const float u = static_cast<float>(rng_state_ & 0xFFFFFFFF) / 4294967296.0F;

        float cumulative = 0.0F;
        for (int32_t i = 0; i < dist.keep; ++i) {
            cumulative += dist.probs[static_cast<std::size_t>(i)];
            if (u < cumulative) {
                result.token_id = dist.indices[static_cast<std::size_t>(i)];
                result.logprob = std::log(std::max(dist.probs[static_cast<std::size_t>(i)],
                                                   std::numeric_limits<float>::min()));
                result.is_eos = params.is_eos(result.token_id);
                return result;
            }
        }

        result.token_id = dist.indices[static_cast<std::size_t>(dist.keep - 1)];
        result.logprob = std::log(std::max(dist.probs[static_cast<std::size_t>(dist.keep - 1)],
                                           std::numeric_limits<float>::min()));
        result.is_eos = params.is_eos(result.token_id);
        return result;
    }
    void reset() override { rng_state_ = initial_seed_; }

  private:
    uint64_t rng_state_;
    uint64_t initial_seed_;
};

} // namespace

Qwen35MoeSamplingParams
qwen35_sampling_params_from_config(const TextGenerationConfig& cfg, int32_t default_eos,
                                   const std::vector<int32_t>& default_eos_ids) {
    Qwen35MoeSamplingParams p;
    p.temperature = cfg.temperature;
    p.top_k = cfg.top_k;
    p.top_p = cfg.top_p;
    p.min_p = cfg.min_p;
    p.seed = cfg.seed;
    p.eos_token_id = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : default_eos;
    p.eos_token_ids =
        cfg.eos_token_id >= 0 ? std::vector<int32_t>{cfg.eos_token_id} : default_eos_ids;
    return p;
}

bool qwen35_is_greedy(const Qwen35MoeSamplingParams& params) {
    const auto top_p = sanitized_top_p(params.top_p);
    const bool default_greedy = params.top_k <= 1 && top_p >= 1.0F - kSamplingEpsilon &&
                                sanitized_min_p(params.min_p) <= 0 && params.seed < 0;
    return default_greedy || greedy_equivalent(params) ||
           (params.top_k == 1 && top_p >= 1.0F - kSamplingEpsilon);
}

std::vector<float> qwen35_probabilities(const float* logits, int32_t vocab_size,
                                        const Qwen35MoeSamplingParams& params) {
    if (vocab_size <= 0 || !logits)
        throw std::invalid_argument("Invalid Qwen3.5-MoE distribution");
    std::vector<float> result(static_cast<std::size_t>(vocab_size), 0);
    const bool default_greedy = params.top_k <= 1 &&
                                sanitized_top_p(params.top_p) >= 1.0F - kSamplingEpsilon &&
                                sanitized_min_p(params.min_p) <= 0.0F && params.seed < 0;
    if (default_greedy || greedy_equivalent(params)) {
        result[argmax_over_logits(logits, vocab_size, {}).token_id] = 1;
    } else {
        const auto dist = build_filtered_distribution(logits, vocab_size, params);
        for (int32_t i = 0; i < dist.keep; ++i)
            result[dist.indices[i]] = dist.probs[i];
    }
    return result;
}

double qwen35_uniform(uint64_t& rng) {
    if (rng == 0)
        rng = 1;
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return static_cast<double>(rng & 0xFFFFFFFFULL) / 4294967296.0;
}
int32_t qwen35_draw_probabilities(const std::vector<float>& probabilities, uint64_t& rng) {
    double mass = 0;
    int32_t last = -1;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const float p = probabilities[i];
        if (!std::isfinite(p) || p < 0)
            throw std::runtime_error("Invalid Qwen3.5-MoE probability");
        mass += p;
        if (p > 0)
            last = static_cast<int32_t>(i);
    }
    if (!(mass > 0))
        throw std::runtime_error("Empty Qwen3.5-MoE probability distribution");
    const double u = qwen35_uniform(rng) * mass;
    double cumulative = 0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (u < cumulative)
            return static_cast<int32_t>(i);
    }
    return last;
}
int32_t qwen35_verify_proposal(int32_t token, const std::vector<float>& target,
                               const std::vector<float>& draft, uint64_t& rng, bool& accepted) {
    if (target.size() != draft.size() || token < 0 ||
        static_cast<std::size_t>(token) >= target.size() || !(draft[token] > 0))
        throw std::invalid_argument("Invalid Qwen3.5-MoE speculative proposal");
    // Normalize both float distributions before p/q, preserving exact target mass.
    const double p_mass = std::accumulate(target.begin(), target.end(), 0.0);
    const double q_mass = std::accumulate(draft.begin(), draft.end(), 0.0);
    if (!(p_mass > 0) || !(q_mass > 0))
        throw std::runtime_error("Empty speculative distribution");
    const double ratio = (target[token] / p_mass) / (draft[token] / q_mass);
    accepted = ratio >= 1 || qwen35_uniform(rng) < ratio;
    if (accepted)
        return token;
    std::vector<float> residual(target.size());
    for (std::size_t i = 0; i < target.size(); ++i)
        residual[i] = static_cast<float>(std::max(0.0, target[i] / p_mass - draft[i] / q_mass));
    return qwen35_draw_probabilities(residual, rng);
}

std::unique_ptr<Qwen35MoeISampler> create_qwen35_sampler(const Qwen35MoeSamplingParams& params) {
    const float top_p = sanitized_top_p(params.top_p);
    const float min_p = sanitized_min_p(params.min_p);
    if (params.top_k <= 1 && top_p >= 1.0F - kSamplingEpsilon && min_p <= 0.0F && params.seed < 0) {
        return std::make_unique<Qwen35MoeGreedySampler>();
    }

    const uint64_t seed = (params.seed >= 0) ? static_cast<uint64_t>(params.seed) : 42ULL;
    return std::make_unique<Qwen35MoeTopKSampler>(seed);
}

} // namespace trtmc
