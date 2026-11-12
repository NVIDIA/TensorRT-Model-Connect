/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace trtmc::vibevoice {

// Standard-normal draws from SplitMix64 and the Box-Muller transform in double
// precision. The stream is fully specified so a reference run can be fed the
// same noise: one draw per request starts from the request's seed.
class NoiseStream {
  public:
    explicit NoiseStream(std::uint64_t seed) noexcept : state_(seed) {}

    float normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return static_cast<float>(spare_);
        }
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 2.0 * kPi * uniform();
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return static_cast<float>(radius * std::cos(angle));
    }

  private:
    static constexpr double kPi = 3.141592653589793;

    std::uint64_t next() noexcept {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t value = state_;
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    // Uniform on (0, 1]: never zero, so the logarithm is finite.
    double uniform() noexcept {
        return static_cast<double>((next() >> 11U) + 1U) * (1.0 / 9007199254740992.0);
    }

    std::uint64_t state_;
    bool has_spare_{false};
    double spare_{0.0};
};

// Round a float to the nearest bfloat16 value (ties to even), as a bfloat16
// tensor stores it. Infinities and NaNs pass through.
inline float round_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U)
        return value;
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

} // namespace trtmc::vibevoice
