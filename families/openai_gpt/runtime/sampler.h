/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::openai_gpt {

struct SamplingParams {
    double temperature{1.0};
    std::int32_t top_k{1}; // 0 disables the top-k limit
    double top_p{1.0};     // 1 disables nucleus filtering
    double min_p{0.0};     // 0 disables the relative-probability floor
    std::int64_t seed{-1}; // < 0 selects the fixed default seed
};

// Host token selection over one row of FP32 logits. Greedy decoding (temperature 0, or
// top_k 1 with top_p 1 and min_p 0) takes the first maximum, as torch.argmax does.
// Sampling keeps the top_k most likely tokens, then the tokens at least min_p times as
// likely as the most likely one, then the smallest prefix whose probability reaches top_p.
class Sampler {
  public:
    // Throws std::invalid_argument on parameters out of range.
    explicit Sampler(const SamplingParams& params);
    // Throws std::runtime_error on a row with a NaN or infinite logit.
    std::int32_t sample(const std::vector<float>& logits);

  private:
    double uniform();

    SamplingParams params_;
    bool greedy_;
    std::uint64_t state_;
};

} // namespace trtmc::openai_gpt
