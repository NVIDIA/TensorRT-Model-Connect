/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cmath>
#include <stdexcept>

namespace trtmc::vitmatte {

inline float normalize_rgb(float value, bool byte) {
    if (!std::isfinite(value) || value < 0 || value > (byte ? 255.0F : 1.0F))
        throw std::invalid_argument("Invalid ViTMatte RGB value");
    // The reference rescales bytes in double precision, stores float32, then
    // normalizes. Combining these operations changes the encoder's input bits.
    const float rescaled =
        byte ? static_cast<float>(static_cast<double>(value) * (1.0 / 255.0)) : value;
    return (rescaled - 0.5F) / 0.5F;
}

} // namespace trtmc::vitmatte
