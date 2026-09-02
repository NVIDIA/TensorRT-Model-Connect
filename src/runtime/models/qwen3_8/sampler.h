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

struct Qwen38SamplingParams {
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    float repetition_penalty{1.0F};
    int32_t seed{-1};
    int32_t eos_token_id{-1};
};

enum class Qwen38LogitsLocation {
    HOST,
    DEVICE,
};

struct Qwen38SampleResult {
    int32_t token_id{0};
    float logprob{0.0F};
    bool is_eos{false};
};

class Qwen38ISampler {
  public:
    virtual ~Qwen38ISampler() = default;
    virtual Qwen38SampleResult sample(const float* logits, int32_t vocab_size,
                                      const Qwen38SamplingParams& params) = 0;
    virtual Qwen38LogitsLocation logits_location() const = 0;
    virtual const char* sampler_type() const = 0;
    virtual void reset() {}
};

Qwen38SamplingParams qwen38_sampling_params_from_config(const GenerateConfig& cfg,
                                                        int32_t default_eos = -1);

// Scale down logits for tokens already present in token_history. Applied to the
// logits before sampling, so both samplers honor it without either needing to
// know the history. A penalty of 1.0 (the default) leaves logits untouched.
void qwen38_apply_repetition_penalty(std::vector<float>& logits, float penalty,
                                     const std::vector<int32_t>& token_history);

std::unique_ptr<Qwen38ISampler> create_qwen38_sampler(const Qwen38SamplingParams& params);

} // namespace trtmc
