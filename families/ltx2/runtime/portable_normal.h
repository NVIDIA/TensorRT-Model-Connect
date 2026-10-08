/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace trtmc::ltx2 {

// std::normal_distribution is implementation-defined: libstdc++ and the MSVC
// STL draw different values from the same std::mt19937 seed, so one seed
// would give a different video per platform. This class reproduces the
// libstdc++ algorithm (generate_canonical<float, 24> plus the Marsaglia polar
// method with one cached value) step by step in float/double arithmetic, so
// other standard libraries draw the same initial latents as Linux builds, up
// to last-bit logf differences between C runtimes.
class LibstdcxxNormalFloat {
  public:
    float operator()(std::mt19937& generator) {
        if (saved_available_) {
            saved_available_ = false;
            return saved_;
        }
        float x = 0.0F;
        float y = 0.0F;
        float r2 = 0.0F;
        do {
            x = static_cast<float>(static_cast<double>(2.0F * canonical(generator)) - 1.0);
            y = static_cast<float>(static_cast<double>(2.0F * canonical(generator)) - 1.0);
            r2 = x * x + y * y;
        } while (r2 > 1.0 || r2 == 0.0);
        // Same float log/sqrt calls as libstdc++. On other C runtimes, logf
        // may differ from glibc in the last bit for a few inputs.
        const float multiplier = std::sqrt(-2.0F * std::log(r2) / r2);
        saved_ = x * multiplier;
        saved_available_ = true;
        return y * multiplier;
    }

  private:
    // generate_canonical<float, 24> over a 32-bit engine: one draw, rounded
    // to float, scaled by 2^-32, clamped below 1.
    static float canonical(std::mt19937& generator) {
        const float value = static_cast<float>(generator() - std::mt19937::min()) / 4294967296.0F;
        return value >= 1.0F ? std::nextafter(1.0F, 0.0F) : value;
    }

    float saved_{0.0F};
    bool saved_available_{false};
};

} // namespace trtmc::ltx2
