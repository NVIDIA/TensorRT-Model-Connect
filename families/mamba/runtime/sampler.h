/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace trtmc {

struct TextGenerationConfig;

struct MambaSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
};

struct MambaSampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class MambaISampler {
  public:
    virtual ~MambaISampler() = default;
    virtual MambaSampleResult sample(const float* logits, int32_t vocab_size,
                                     const MambaSamplingParams& params) = 0;
    virtual void reset() {}
};

MambaSamplingParams mamba_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                      int32_t default_eos = -1);

std::unique_ptr<MambaISampler> create_mamba_sampler(const MambaSamplingParams& params);

} // namespace trtmc
