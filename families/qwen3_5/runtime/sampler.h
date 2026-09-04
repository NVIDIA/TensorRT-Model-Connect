/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace trtmc {

struct TextGenerationConfig;

struct Qwen35SamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
};

struct Qwen35SampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class Qwen35ISampler {
  public:
    virtual ~Qwen35ISampler() = default;
    virtual Qwen35SampleResult sample(const float* logits, int32_t vocab_size,
                                      const Qwen35SamplingParams& params) = 0;
    virtual void reset() {}
};

Qwen35SamplingParams qwen35_sampling_params_from_config(const TextGenerationConfig& cfg,
                                                        int32_t default_eos = -1);

std::unique_ptr<Qwen35ISampler> create_qwen35_sampler(const Qwen35SamplingParams& params);

} // namespace trtmc
