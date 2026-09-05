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

} // namespace trtmc
