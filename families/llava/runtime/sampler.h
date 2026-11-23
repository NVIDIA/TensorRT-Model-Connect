/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Family-owned host token sampling.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

/// Sampling parameters -- controls token selection behavior.
struct LlavaSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
    std::vector<int32_t> additional_eos_token_ids;

    bool is_eos(int32_t token) const {
        return token == eos_token_id ||
               std::find(additional_eos_token_ids.begin(), additional_eos_token_ids.end(), token) !=
                   additional_eos_token_ids.end();
    }
};

/// Token selection result.
struct LlavaSampleResult {
    int32_t token_id{0};
    float logprob{0.0f}; // log-probability of selected token (informational)
    bool is_eos{false};  // true if token_id matches eos_token_id
};

/// llava-owned sampler interface.
class LlavaISampler {
  public:
    virtual ~LlavaISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] in host memory.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual LlavaSampleResult sample(const float* logits, int32_t vocab_size,
                                     const LlavaSamplingParams& params) = 0;

    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build LlavaSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/pipeline.h

LlavaSamplingParams llava_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                      int32_t default_eos = -1);

/// Factory: create sampler from LlavaSamplingParams.
/// - top_k <= 1 && top_p/min_p disabled && seed == -1 => GreedySampler
/// - otherwise => TopKSampler
std::unique_ptr<LlavaISampler> create_llava_sampler(const LlavaSamplingParams& params);

} // namespace trtmc
