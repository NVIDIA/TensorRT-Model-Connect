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
struct OptSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
};

/// Token selection result.
struct OptSampleResult {
    int32_t token_id{0};
    bool is_eos{false}; // true if token_id matches eos_token_id
};

/// opt-owned sampler interface.
class OptISampler {
  public:
    virtual ~OptISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] on host.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual OptSampleResult sample(const float* logits, int32_t vocab_size,
                                   const OptSamplingParams& params) = 0;
    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build OptSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/task.h

OptSamplingParams opt_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                  int32_t default_eos = -1);

/// Create the exact host sampler selected by the request.
std::unique_ptr<OptISampler> create_opt_sampler(const OptSamplingParams& params);

} // namespace trtmc
