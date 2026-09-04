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
struct GemmaSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
    std::vector<int32_t> eos_token_ids;
};

/// Token selection result.
struct GemmaSampleResult {
    int32_t token_id{0};
    bool is_eos{false}; // true if token_id matches eos_token_id
};

/// gemma-owned sampler interface.
class GemmaISampler {
  public:
    virtual ~GemmaISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] on host.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual GemmaSampleResult sample(const float* logits, int32_t vocab_size,
                                     const GemmaSamplingParams& params) = 0;
    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build GemmaSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/task.h

GemmaSamplingParams
gemma_sampling_params_from_config(const TextGenerationConfig& cfg,
                                  const std::vector<int32_t>& default_eos_token_ids);
GemmaSamplingParams gemma_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                      int32_t default_eos = -1);

/// Return true when token_id matches any effective EOS token.
bool gemma_is_eos_token(const GemmaSamplingParams& params, int32_t token_id);

/// Create the exact host sampler selected by the request.
std::unique_ptr<GemmaISampler> create_gemma_sampler(const GemmaSamplingParams& params);

} // namespace trtmc
