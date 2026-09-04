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

struct NemotronHSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    int32_t seed{-1};
    std::vector<int32_t> eos_token_ids{};
};

struct NemotronHSampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class NemotronHISampler {
  public:
    virtual ~NemotronHISampler() = default;
    virtual NemotronHSampleResult sample(const float* logits, int32_t vocab_size,
                                         const NemotronHSamplingParams& params) = 0;
    virtual void reset() {}
};

NemotronHSamplingParams
nemotron_h_sampling_params_from_config(const TextGenerationConfig& cfg,
                                       const std::vector<int32_t>& default_eos_token_ids);

std::unique_ptr<NemotronHISampler> create_nemotron_h_sampler(const NemotronHSamplingParams& params);

} // namespace trtmc
