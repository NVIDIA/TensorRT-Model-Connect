/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::minicpmv4_6 {

struct SamplingOptions {
    float temperature{1.0F};
    std::int32_t top_k{1};
    float top_p{1.0F};
    std::int64_t seed{-1}; // -1: a fixed default seed
};

// Sampling options from a request's values, checked before they are narrowed to FP32: a
// negative or nonfinite value, a positive temperature or top_p that FP32 rounds to zero, and a
// top_p above one are rejected.
SamplingOptions sampling_options(double temperature, std::int64_t top_k, double top_p,
                                 std::int64_t seed);

// Greedy argmax when top_k == 1 (the default) or the temperature is zero; otherwise temperature,
// top-k (0 keeps the whole vocabulary) and top-p filtering with a seeded xorshift draw.
class Sampler {
  public:
    explicit Sampler(const SamplingOptions& options);
    // A token id below `limit` (the tokenizer's vocabulary; an embedding table may have more rows).
    std::int32_t sample(const std::vector<float>& logits, std::int32_t limit);

  private:
    SamplingOptions options_;
    bool greedy_{true};
    std::uint64_t state_{1};
};

} // namespace trtmc::minicpmv4_6
