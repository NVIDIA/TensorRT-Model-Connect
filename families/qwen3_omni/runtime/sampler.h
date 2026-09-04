/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc::qwen3_omni {

// Request-owned sampler for the 15 residual codec groups. Construct it once
// from the request seed and reuse it across every group and audio frame.
class ResidualCodeSampler final {
  public:
    explicit ResidualCodeSampler(std::uint64_t seed);
    ~ResidualCodeSampler();

    ResidualCodeSampler(const ResidualCodeSampler&) = delete;
    ResidualCodeSampler& operator=(const ResidualCodeSampler&) = delete;

    // Apply the checkpoint's fixed sampling policy: top-k=50, top-p=0.8,
    // softmax, then PyTorch CUDA's no-replacement n=1 exponential race.
    // Invalid logits fail closed.
    std::int32_t sample(const float* logits, std::size_t count);

    std::uint64_t draws() const { return draws_; }

  private:
    std::uint64_t seed_{0};
    std::uint64_t draws_{0};
    float* device_probabilities_{nullptr};
    std::int32_t* device_token_{nullptr};
};

} // namespace trtmc::qwen3_omni
