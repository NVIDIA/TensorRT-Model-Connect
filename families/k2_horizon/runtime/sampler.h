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

struct K2HorizonSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
    std::vector<int32_t> eos_token_ids;
};

struct K2HorizonSampleResult {
    int32_t token_id{0};
    bool is_eos{false};
};

class K2HorizonISampler {
  public:
    virtual ~K2HorizonISampler() = default;
    virtual K2HorizonSampleResult sample(const float* logits, int32_t vocab_size,
                                         const K2HorizonSamplingParams& params) = 0;
    virtual const char* sampler_type() const = 0;
    virtual void reset() {}
};

K2HorizonSamplingParams
k2_horizon_sampling_params_from_config(const TextGenerationConfig& cfg,
                                       const std::vector<int32_t>& default_eos_token_ids);
K2HorizonSamplingParams k2_horizon_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                               int32_t default_eos = -1);
bool k2_horizon_is_eos_token(const K2HorizonSamplingParams& params, int32_t token_id);
void k2_horizon_validate_sampling_params(const K2HorizonSamplingParams& params, int32_t vocab_size);
std::unique_ptr<K2HorizonISampler> create_k2_horizon_sampler(const K2HorizonSamplingParams& params);

} // namespace trtmc
