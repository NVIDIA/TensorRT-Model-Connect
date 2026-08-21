/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct GenerateConfig;

struct Lfm2MoeSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    std::vector<int32_t> eos_token_ids;
};

struct Lfm2MoeSampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class Lfm2MoeISampler {
  public:
    virtual ~Lfm2MoeISampler() = default;
    virtual Lfm2MoeSampleResult sample(const float* logits, int32_t vocab_size,
                                    const Lfm2MoeSamplingParams& params,
                                    const std::vector<int32_t>& token_history) = 0;
    virtual const char* sampler_type() const = 0;
    virtual void reset() {}
};

Lfm2MoeSamplingParams
lfm2_moe_sampling_params_from_config(const GenerateConfig& cfg,
                                 const std::vector<int32_t>& default_eos_token_ids);

// HF repetition penalty is sign-aware and is applied once per unique token in
// the complete prompt+generated history, before temperature or probability
// filtering.
std::vector<float> lfm2_moe_apply_repetition_penalty(const float* logits, int32_t vocab_size,
                                                 float penalty,
                                                 const std::vector<int32_t>& token_history);

int32_t lfm2_moe_resolve_top_k(const Lfm2MoeSamplingParams& params);
bool lfm2_moe_is_eos_token(const Lfm2MoeSamplingParams& params, int32_t token_id);
std::unique_ptr<Lfm2MoeISampler> create_lfm2_moe_sampler(const Lfm2MoeSamplingParams& params);

} // namespace trtmc
