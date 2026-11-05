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

struct Lfm3SamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    std::vector<int32_t> eos_token_ids;
};

struct Lfm3SampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class Lfm3ISampler {
  public:
    virtual ~Lfm3ISampler() = default;
    virtual Lfm3SampleResult sample(const float* logits, int32_t vocab_size,
                                    const Lfm3SamplingParams& params,
                                    const std::vector<int32_t>& token_history) = 0;
    virtual void reset() {}
};

Lfm3SamplingParams
lfm3_sampling_params_from_config(const TextGenerationConfig& cfg,
                                 const std::vector<int32_t>& default_eos_token_ids);

std::vector<float> lfm3_apply_repetition_penalty(const float* logits, int32_t vocab_size,
                                                 float penalty,
                                                 const std::vector<int32_t>& token_history);

int32_t lfm3_resolve_top_k(const Lfm3SamplingParams& params);
bool lfm3_is_eos_token(const Lfm3SamplingParams& params, int32_t token_id);
std::unique_ptr<Lfm3ISampler> create_lfm3_sampler(const Lfm3SamplingParams& params);

} // namespace trtmc
