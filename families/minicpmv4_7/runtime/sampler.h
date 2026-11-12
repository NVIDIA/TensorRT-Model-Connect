/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace trtmc::minicpmv4_7 {

struct SamplingConfig {
    double temperature{1.0};
    int64_t top_k{1};
    double top_p{1.0};
    int64_t seed{-1};
};

// Greedy (first maximum) when top_k == 1 or temperature == 0; otherwise temperature, top-k and
// nucleus filtering followed by one draw from a seeded generator.
class Sampler {
  public:
    explicit Sampler(const SamplingConfig& config);
    int32_t sample(const std::vector<float>& logits);

  private:
    SamplingConfig config_;
    std::mt19937_64 generator_;
};

} // namespace trtmc::minicpmv4_7
