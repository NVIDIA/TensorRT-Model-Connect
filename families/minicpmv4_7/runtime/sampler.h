/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace trtmc::minicpmv4_7 {

inline constexpr int64_t kDefaultMaxNewTokens = 128;

// Generated-token budget for a prompt of `prompt_tokens` ids in a decoder of `capacity` cache rows:
// the requested value, or up to kDefaultMaxNewTokens within the room the prompt leaves. The last
// generated token is never fed back, so it needs no cache row. Throws when the budget does not fit.
int64_t generation_budget(std::optional<int64_t> requested, int64_t capacity,
                          int64_t prompt_tokens);

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
