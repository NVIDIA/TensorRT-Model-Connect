/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::apertus {

struct SamplingParams {
    float temperature{1.0F};
    std::int32_t top_k{1}; // <= 0 disables the top-k limit
    float top_p{1.0F};     // 1 disables nucleus filtering
    float min_p{0.0F};     // 0 disables the relative-probability floor
    std::int64_t seed{-1}; // < 0 selects the fixed default seed
};

// Host token selection over one row of FP32 logits. Greedy decoding (top_k <= 1
// with top_p = 1 and min_p = 0, or temperature 0) takes the first maximum.
class Sampler {
  public:
    explicit Sampler(const SamplingParams& params);
    std::int32_t sample(const std::vector<float>& logits);

  private:
    float uniform();

    SamplingParams params_;
    bool greedy_;
    std::uint64_t state_;
};

} // namespace trtmc::apertus
