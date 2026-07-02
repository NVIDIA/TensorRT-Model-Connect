/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace trtmc {

struct GenerateConfig;

struct RwkvSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
};

enum class RwkvLogitsLocation {
    HOST,
    DEVICE,
};

struct RwkvSampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class RwkvISampler {
  public:
    virtual ~RwkvISampler() = default;
    virtual RwkvSampleResult sample(const float* logits, int32_t vocab_size,
                                    const RwkvSamplingParams& params) = 0;
    virtual RwkvLogitsLocation logits_location() const = 0;
    virtual const char* sampler_type() const = 0;
    virtual void reset() {}
};

RwkvSamplingParams rwkv_sampling_params_from_config(const GenerateConfig& cfg,
                                                    int32_t default_eos = -1);

std::unique_ptr<RwkvISampler> create_rwkv_sampler(const RwkvSamplingParams& params);

} // namespace trtmc
