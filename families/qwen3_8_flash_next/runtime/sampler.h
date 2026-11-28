/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct TextGenerationConfig;

struct FlashNextSamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
    std::vector<int32_t> eos_token_ids{};

    bool is_eos(int32_t token) const {
        return eos_token_ids.empty() ? token == eos_token_id
                                     : std::find(eos_token_ids.begin(), eos_token_ids.end(),
                                                 token) != eos_token_ids.end();
    }
};

struct FlashNextSampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class FlashNextISampler {
  public:
    virtual ~FlashNextISampler() = default;
    virtual FlashNextSampleResult sample(const float* logits, int32_t vocab_size,
                                         const FlashNextSamplingParams& params) = 0;
    virtual void reset() {}
};

FlashNextSamplingParams
flash_next_sampling_params_from_config(const TextGenerationConfig& cfg, int32_t default_eos = -1,
                                       const std::vector<int32_t>& default_eos_ids = {});

std::unique_ptr<FlashNextISampler> create_flash_next_sampler(const FlashNextSamplingParams& params);

} // namespace trtmc
