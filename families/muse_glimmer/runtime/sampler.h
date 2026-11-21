/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Family-owned host samplers for autoregressive generation.

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

/// Sampling parameters -- controls token selection behavior.
struct MuseSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
};

/// Token selection result.
struct MuseSampleResult {
    int32_t token_id{0};
    bool is_eos{false}; // true if token_id matches eos_token_id
};

/// muse-owned sampler interface.
class MuseISampler {
  public:
    virtual ~MuseISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] on host.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual MuseSampleResult sample(const float* logits, int32_t vocab_size,
                                    const MuseSamplingParams& params) = 0;
    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build MuseSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/task.h

MuseSamplingParams muse_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                    int32_t default_eos = -1);

/// Whether the request selects the same deterministic argmax distribution.
bool muse_is_greedy(const MuseSamplingParams& params);

/// Create the exact host sampler selected by the request.
std::unique_ptr<MuseISampler> create_muse_sampler(const MuseSamplingParams& params);

/// Family-local proposal distribution and exact p/q rejection correction.
std::vector<float> muse_probabilities(const float* logits, int32_t vocab_size,
                                      const MuseSamplingParams& params);
double muse_uniform(uint64_t& rng);
int32_t muse_draw_probabilities(const std::vector<float>& probabilities, uint64_t& rng);
int32_t muse_verify_proposal(int32_t token, const std::vector<float>& target,
                             const std::vector<float>& draft, uint64_t& rng, bool& accepted);

} // namespace trtmc
