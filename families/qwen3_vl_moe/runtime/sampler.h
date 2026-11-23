/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Family-owned host token sampling.

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

/// Sampling parameters -- controls token selection behavior.
struct Qwen3VlMoeSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
    std::vector<int32_t> eos_token_ids;
};

/// Token selection result.
struct Qwen3VlMoeSampleResult {
    int32_t token_id{0};
    float logprob{0.0f}; // log-probability of selected token (informational)
    bool is_eos{false};  // true if token_id matches eos_token_id
};

/// qwen3_vl_moe-owned sampler interface.
class Qwen3VlMoeISampler {
  public:
    virtual ~Qwen3VlMoeISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] in host memory.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual Qwen3VlMoeSampleResult sample(const float* logits, int32_t vocab_size,
                                          const Qwen3VlMoeSamplingParams& params) = 0;

    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build Qwen3VlMoeSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/pipeline.h

Qwen3VlMoeSamplingParams
qwen3_vl_moe_sampling_params_from_config(const TextGenerationConfig& cfg,
                                         const std::vector<int32_t>& default_eos_token_ids);
Qwen3VlMoeSamplingParams qwen3_vl_moe_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                                  int32_t default_eos = -1);

/// Return true when token_id matches any effective EOS token.
bool qwen3_vl_moe_is_eos_token(const Qwen3VlMoeSamplingParams& params, int32_t token_id);

/// Factory: create sampler from Qwen3VlMoeSamplingParams.
/// - top_k <= 1 && top_p/min_p disabled && seed == -1 => GreedySampler
/// - otherwise => TopKSampler
std::unique_ptr<Qwen3VlMoeISampler>
create_qwen3_vl_moe_sampler(const Qwen3VlMoeSamplingParams& params);

} // namespace trtmc
