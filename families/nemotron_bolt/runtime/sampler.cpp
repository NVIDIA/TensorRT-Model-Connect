/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NemotronBoltISampler implementations: GreedySampler, TopKSampler, and factory.
//
// GreedySampler uses std::max_element, preserving the decoder's first-max
// tie-breaking rule and deterministic token sequences.
//
// TopKSampler handles temperature, top-k, top-p, and min-p sampling on host
// logits, with internal xorshift64 RNG state.

#include "families/nemotron_bolt/runtime/sampler.h"

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace trtmc {

// Host argmax uses the first maximum, matching deterministic greedy decoding.
static NemotronBoltSampleResult argmax_over_logits(const float* logits, int32_t vocab_size,
                                                   int32_t eos_token_id) {
    NemotronBoltSampleResult result;
    const float* best = logits;
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > *best)
            best = logits + i;
    }
    result.token_id = static_cast<int32_t>(best - logits);
    result.is_eos = (result.token_id == eos_token_id);
    return result;
}

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probs;
    int32_t keep{0};
};

static constexpr float kSamplingEpsilon = 1e-6F;

static float sanitized_temperature(float temperature) {
    if (!std::isfinite(temperature))
        return 1.0F;
    return std::max(temperature, 0.0F);
}

static float sanitized_top_p(float top_p) {
    if (!std::isfinite(top_p))
        return 1.0F;
    return std::min(std::max(top_p, 0.0F), 1.0F);
}

static float sanitized_min_p(float min_p) {
    if (!std::isfinite(min_p))
        return 0.0F;
    return std::min(std::max(min_p, 0.0F), 1.0F);
}

static bool top_p_enabled(float top_p) {
    return top_p > 0.0F && top_p < 1.0F - kSamplingEpsilon;
}

static bool greedy_equivalent(const NemotronBoltSamplingParams& params) {
    const float temperature = sanitized_temperature(params.temperature);
    const float top_p = sanitized_top_p(params.top_p);
    return temperature < kSamplingEpsilon || top_p <= 0.0F;
}

static void topk_indices_and_softmax(FilteredDistribution& dist, const float* logits, int32_t n,
                                     int32_t k, float temperature) {
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

static int32_t apply_min_p(const FilteredDistribution& dist, int32_t k, float min_p) {
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

static int32_t apply_top_p(const FilteredDistribution& dist, int32_t keep, float top_p) {
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

static void renormalize_kept_prefix(FilteredDistribution& dist, int32_t keep) {
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

static FilteredDistribution build_filtered_distribution(const float* logits, int32_t vocab_size,
                                                        const NemotronBoltSamplingParams& params) {
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

// ─────────────────────────────────────────────────────────────
// GreedySampler: deterministic argmax (identical to select_argmax_token)
// ─────────────────────────────────────────────────────────────

class GreedySampler final : public NemotronBoltISampler {
  public:
    NemotronBoltSampleResult sample(const float* logits, int32_t vocab_size,
                                    const NemotronBoltSamplingParams& params) override {
        if (vocab_size <= 0 || logits == nullptr) {
            NemotronBoltSampleResult result;
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        return argmax_over_logits(logits, vocab_size, params.eos_token_id);
    }
};

// ─────────────────────────────────────────────────────────────
// TopKSampler: temperature-scaled top-k with xorshift64 RNG
// (identical logic to sample_token_topk)
// ─────────────────────────────────────────────────────────────

class TopKSampler final : public NemotronBoltISampler {
  public:
    explicit TopKSampler(uint64_t initial_seed)
        : rng_state_(initial_seed == 0 ? 1 : initial_seed),
          initial_seed_(initial_seed == 0 ? 1 : initial_seed) {}

    NemotronBoltSampleResult sample(const float* logits, int32_t vocab_size,
                                    const NemotronBoltSamplingParams& params) override {
        NemotronBoltSampleResult result;

        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        if (greedy_equivalent(params))
            return argmax_over_logits(logits, vocab_size, params.eos_token_id);

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);

        // xorshift64 random number generation
        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        float u = static_cast<float>(rng_state_ & 0xFFFFFFFF) / 4294967296.0F;

        // Sample from cumulative distribution
        float cumulative = 0.0F;
        for (int32_t i = 0; i < dist.keep; ++i) {
            cumulative += dist.probs[static_cast<std::size_t>(i)];
            if (u < cumulative) {
                result.token_id = dist.indices[static_cast<std::size_t>(i)];
                result.is_eos = (result.token_id == params.eos_token_id);
                return result;
            }
        }

        result.token_id = dist.indices[static_cast<std::size_t>(dist.keep - 1)];
        result.is_eos = (result.token_id == params.eos_token_id);
        return result;
    }

    void reset() override { rng_state_ = initial_seed_; }

  private:
    uint64_t rng_state_;
    uint64_t initial_seed_;
};

// ─────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────

NemotronBoltSamplingParams
nemotron_bolt_sampling_params_from_config(const TextGenerationConfig& cfg, int32_t default_eos) {
    NemotronBoltSamplingParams p;
    p.temperature = cfg.temperature;
    p.top_k = cfg.top_k;
    p.top_p = cfg.top_p;
    p.min_p = cfg.min_p;
    p.seed = cfg.seed;
    p.eos_token_id = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : default_eos;
    return p;
}

std::unique_ptr<NemotronBoltISampler>
create_nemotron_bolt_sampler(const NemotronBoltSamplingParams& params) {
    // Greedy when sampling is fully disabled and no explicit random seed is set.
    const float top_p = sanitized_top_p(params.top_p);
    const float min_p = sanitized_min_p(params.min_p);
    if (params.top_k <= 1 && top_p >= 1.0F - kSamplingEpsilon && min_p <= 0.0F && params.seed < 0) {
        return std::make_unique<GreedySampler>();
    }

    uint64_t seed = (params.seed >= 0) ? static_cast<uint64_t>(params.seed)
                                       : 42ULL; // deterministic default for reproducibility

    // TopK sampler with xorshift64 RNG
    return std::make_unique<TopKSampler>(seed);
}

BoltVerificationSampler::BoltVerificationSampler(NemotronBoltSamplingParams params)
    : params_(params), rng_(params.seed < 0 ? 42 : std::max<uint64_t>(params.seed, 1)) {}
double BoltVerificationSampler::uniform() {
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 7;
    rng_ ^= rng_ << 17;
    return double(rng_ >> 11) * 0x1.0p-53;
}
int32_t BoltVerificationSampler::draw(const std::vector<double>& weights) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total > 0) || !std::isfinite(total))
        throw std::runtime_error("Bolt sampling has no probability mass");
    const double u = uniform() * total;
    double sum = 0;
    int last = -1;
    for (size_t i = 0; i < weights.size(); ++i)
        if (weights[i] > 0) {
            last = int(i);
            sum += weights[i];
            if (u < sum)
                return last;
        }
    return last;
}
std::vector<double> BoltVerificationSampler::distribution(const std::vector<float>& logits) const {
    if (logits.empty())
        throw std::invalid_argument("No Bolt target logits");
    std::vector<double> p(logits.size());
    const bool deterministic = params_.top_k <= 1 &&
                               sanitized_top_p(params_.top_p) >= 1 - kSamplingEpsilon &&
                               sanitized_min_p(params_.min_p) <= 0 && params_.seed < 0;
    if (greedy_equivalent(params_) || deterministic) {
        p[std::max_element(logits.begin(), logits.end()) - logits.begin()] = 1;
    } else {
        auto d = build_filtered_distribution(logits.data(), int(logits.size()), params_);
        double mass = 0;
        for (int j = 0; j < d.keep; ++j)
            mass += d.probs[j];
        for (int j = 0; j < d.keep; ++j)
            p[d.indices[j]] = d.probs[j] / mass;
    }
    return p;
}
int32_t BoltVerificationSampler::target(const std::vector<float>& logits) {
    return draw(distribution(logits));
}
BoltDraftChoice BoltVerificationSampler::propose(const int32_t* ids, const float* scores) {
    BoltDraftChoice c;
    std::copy_n(ids, 8, c.ids.begin());
    if (!std::all_of(scores, scores + 8, [](float f) { return std::isfinite(f); }))
        throw std::runtime_error("Nonfinite Bolt transition scores");
    const auto max = std::max_element(scores, scores + 8);
    const bool greedy = params_.temperature < 1e-3f || params_.top_p <= 0 ||
                        (params_.top_k == 1 && !top_p_enabled(params_.top_p)) ||
                        (params_.top_k <= 1 && params_.top_p >= 1 - kSamplingEpsilon &&
                         params_.min_p <= 0 && params_.seed < 0);
    if (greedy) {
        c.column = int(max - scores);
        c.probability[c.column] = 1;
        return c;
    }
    double sum = 0;
    for (int i = 0; i < 8; ++i) {
        c.probability[i] = std::exp((double(scores[i]) - *max) / params_.temperature);
        sum += c.probability[i];
    }
    for (double& v : c.probability)
        v /= sum;
    c.column = draw({c.probability.begin(), c.probability.end()});
    return c;
}
std::pair<int32_t, bool> BoltVerificationSampler::verify(const std::vector<float>& logits,
                                                         const BoltDraftChoice& c) {
    auto p = distribution(logits);
    const int id = c.ids.at(c.column);
    const double q = c.probability.at(c.column);
    if (!(q > 0))
        throw std::runtime_error("Bolt proposed with zero probability");
    const double ratio = p.at(id) / q;
    if (ratio >= 1 || uniform() < ratio)
        return {id, true};
    for (int j = 0; j < 8; ++j)
        p.at(c.ids[j]) = std::max(0., p.at(c.ids[j]) - c.probability[j]);
    return {draw(p), false};
}
} // namespace trtmc
