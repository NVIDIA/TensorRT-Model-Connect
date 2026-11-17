/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_video_diffusion/runtime/torch_random.h"

#include <cmath>
#include <stdexcept>

namespace trtmc::stable_video_diffusion {

namespace {

constexpr std::size_t kBlock = 16;
constexpr std::size_t kHalfBlock = kBlock / 2;
constexpr double kPi = 3.14159265358979323846;

void box_muller_block(float* values) {
    for (std::size_t pair = 0; pair < kHalfBlock; ++pair) {
        const float u1 = 1.0F - values[pair]; // [0, 1) -> (0, 1] for the logarithm.
        const float u2 = values[pair + kHalfBlock];
        const float radius = std::sqrt(-2.0F * std::log(u1));
        const auto theta = static_cast<float>(2.0 * kPi * static_cast<double>(u2));
        values[pair] = radius * std::cos(theta);
        values[pair + kHalfBlock] = radius * std::sin(theta);
    }
}

} // namespace

float TorchCpuGenerator::uniform() {
    // The low 24 bits of one 32-bit draw, scaled to [0, 1).
    constexpr std::uint32_t kMask = (1U << 24) - 1U;
    constexpr float kDivisor = 1.0F / static_cast<float>(1U << 24);
    return static_cast<float>(engine_() & kMask) * kDivisor;
}

void TorchCpuGenerator::normal(float* values, std::size_t count) {
    if (values == nullptr || count < kBlock)
        throw std::invalid_argument("torch-compatible normal noise needs at least 16 values");
    for (std::size_t index = 0; index < count; ++index)
        values[index] = uniform();
    for (std::size_t index = 0; index + kBlock <= count; index += kBlock)
        box_muller_block(values + index);
    if (count % kBlock != 0) {
        float* tail = values + count - kBlock;
        for (std::size_t index = 0; index < kBlock; ++index)
            tail[index] = uniform();
        box_muller_block(tail);
    }
}

} // namespace trtmc::stable_video_diffusion
