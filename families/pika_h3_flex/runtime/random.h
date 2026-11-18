/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::pika_h3_flex {
// The checkpoint's sampler uses the ATen CPU generator for every noise draw.
// Delegating its Gaussian transform also preserves its floating-point rounding.
class Random {
  public:
    explicit Random(uint32_t seed);
    ~Random();
    std::vector<float> normal(std::size_t count);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::pika_h3_flex
