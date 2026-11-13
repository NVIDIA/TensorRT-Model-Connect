/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace trtmc::pika_h3_flex {
// CPU-generator stream: MT19937 low-24-bit uniforms and 16-value Box–Muller
// groups. Every image/audio draw in the declared 480p profile is a multiple
// of 16. No GPU or CPU instruction-set selection changes this algorithm.
class Random {
  public:
    explicit Random(uint32_t seed) : engine_(seed) {}
    std::vector<float> normal(std::size_t count) {
        if (!count || count % 16)
            throw std::invalid_argument("Pika noise draws require groups of 16");
        std::vector<float> result(count);
        for (auto& value : result)
            value = static_cast<float>(engine_() & 0xffffffU) * (1.0F / 16777216);
        constexpr float two_pi = 6.2831853071795864769F;
        for (std::size_t start = 0; start < count; start += 16) {
            for (int index = 0; index < 8; ++index) {
                const float radius = std::sqrt(-2.0F * std::log(1.0F - result[start + index]));
                const float theta = two_pi * result[start + index + 8];
                result[start + index] = radius * std::cos(theta);
                result[start + index + 8] = radius * std::sin(theta);
            }
        }
        return result;
    }

  private:
    std::mt19937 engine_;
};
} // namespace trtmc::pika_h3_flex
