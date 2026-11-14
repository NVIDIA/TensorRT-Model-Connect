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

struct Qwen35SamplingParams {
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

Qwen35SamplingParams
qwen35_sampling_params_from_config(const TextGenerationConfig& cfg, int32_t default_eos = -1,
                                   const std::vector<int32_t>& default_eos_ids = {});

// The same target filters are used for paired proposal and rejection sampling.
bool qwen35_is_greedy(const Qwen35SamplingParams& params);
std::vector<float> qwen35_probabilities(const float* logits, int32_t vocab_size,
                                        const Qwen35SamplingParams& params);
double qwen35_uniform(uint64_t& rng);
int32_t qwen35_draw_probabilities(const std::vector<float>& probabilities, uint64_t& rng);
int32_t qwen35_verify_proposal(int32_t token, const std::vector<float>& target,
                               const std::vector<float>& draft, uint64_t& rng, bool& accepted);

std::unique_ptr<Qwen35ISampler> create_qwen35_sampler(const Qwen35SamplingParams& params);

} // namespace trtmc
