/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct TextGenerationConfig;
class ITrtModule;

struct Lfm2SamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    std::vector<int32_t> eos_token_ids;
};

struct Lfm2SampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

// Candidates are ordered by descending processed score, then token ID. Keeping
// the distribution separate from the RNG allows reference comparisons to use
// the same uniform draw without assuming matching framework RNG algorithms.
struct Lfm2SamplingDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probabilities;
};

Lfm2SamplingDistribution lfm2_sampling_distribution(const float* logits, int32_t vocab_size,
                                                    const Lfm2SamplingParams& params,
                                                    const std::vector<int32_t>& token_history);
Lfm2SampleResult lfm2_sample_from_distribution(const Lfm2SamplingDistribution& distribution,
                                               float uniform, const Lfm2SamplingParams& params);

class Lfm2ISampler {
  public:
    virtual ~Lfm2ISampler() = default;
    virtual Lfm2SampleResult sample(const float* logits, int32_t vocab_size,
                                    const Lfm2SamplingParams& params,
                                    const std::vector<int32_t>& token_history) = 0;
    virtual void reset() {}
};

Lfm2SamplingParams
lfm2_sampling_params_from_config(const TextGenerationConfig& cfg,
                                 const std::vector<int32_t>& default_eos_token_ids);

// HF repetition penalty is sign-aware and is applied once per unique token in
// the complete prompt+generated history, before temperature or probability
// filtering.
std::vector<float> lfm2_apply_repetition_penalty(const float* logits, int32_t vocab_size,
                                                 float penalty,
                                                 const std::vector<int32_t>& token_history);

int32_t lfm2_resolve_top_k(const Lfm2SamplingParams& params);
bool lfm2_is_eos_token(const Lfm2SamplingParams& params, int32_t token_id);
std::unique_ptr<Lfm2ISampler> create_lfm2_sampler(const Lfm2SamplingParams& params);
bool lfm2_sampling_uses_distribution(const Lfm2SamplingParams& params);
// The module is pipeline-owned; each returned sampler owns only its request's
// seed/offset and preserves the same CPU probability processors.
void lfm2_validate_sampler_module(const ITrtModule& module);
std::unique_ptr<Lfm2ISampler> create_lfm2_sampler(const Lfm2SamplingParams& params,
                                                  ITrtModule& module);

} // namespace trtmc
