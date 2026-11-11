/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host token samplers: greedy (first maximum) and temperature / top-k / top-p /
// min-p sampling with a seeded xorshift64 generator.

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::laguna {

struct SamplingParams {
    float temperature{1.0F};
    std::int32_t top_k{1}; // 1 = greedy unless top_p is active; <= 0 = no top-k limit
    float top_p{1.0F};     // 1.0 = disabled
    float min_p{0.0F};     // 0.0 = disabled
    std::int32_t seed{-1}; // -1 with greedy settings selects the greedy sampler
    std::vector<std::int32_t> eos_token_ids;
};

struct SampleResult {
    std::int32_t token_id{0};
    bool is_eos{false};
};

class Sampler {
  public:
    virtual ~Sampler() = default;
    virtual SampleResult sample(const float* logits, std::int32_t vocab_size,
                                const SamplingParams& params) = 0;
};

bool is_eos_token(const SamplingParams& params, std::int32_t token_id);
std::unique_ptr<Sampler> create_sampler(const SamplingParams& params);

} // namespace trtmc::laguna
