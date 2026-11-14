/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio/runtime/noise.h"

#include <cmath>

namespace trtmc::stable_audio {
namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;
constexpr double kTwoPi = 6.283185307179586;
constexpr double kUnit = 1.0 / 9007199254740992.0; // 2^-53

} // namespace

std::uint64_t mix64(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

void fill_gaussian(std::uint64_t seed, std::uint64_t stream, float* out, std::size_t count) {
    const std::uint64_t key = mix64(seed + kGolden * (stream + 1U));
    for (std::size_t pair = 0; 2 * pair < count; ++pair) {
        const std::uint64_t first = mix64(key + kGolden * (2U * pair + 1U));
        const std::uint64_t second = mix64(key + kGolden * (2U * pair + 2U));
        const double u1 = static_cast<double>((first >> 11U) + 1U) * kUnit;
        const double u2 = static_cast<double>(second >> 11U) * kUnit;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = kTwoPi * u2;
        out[2 * pair] = static_cast<float>(radius * std::cos(angle));
        if (2 * pair + 1 < count)
            out[2 * pair + 1] = static_cast<float>(radius * std::sin(angle));
    }
}

} // namespace trtmc::stable_audio
