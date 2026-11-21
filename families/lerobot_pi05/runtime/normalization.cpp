/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_pi05/runtime/normalization.h"

namespace trtmc::lerobot_pi05 {

float normalize_value(const Normalization& normalization, std::size_t index, float value) {
    if (normalization.identity())
        return value;
    const float offset = normalization.offset[index];
    const float scale = normalization.scale[index];
    if (normalization.mode == NormalizationMode::kMeanStd)
        return (value - offset) / (scale + normalization.eps);
    return 2.0F * (value - offset) / scale - 1.0F;
}

float unnormalize_value(const Normalization& normalization, std::size_t index, float value) {
    if (normalization.identity())
        return value;
    const float offset = normalization.offset[index];
    const float scale = normalization.scale[index];
    if (normalization.mode == NormalizationMode::kMeanStd)
        return value * scale + offset;
    return (value + 1.0F) * scale / 2.0F + offset;
}

} // namespace trtmc::lerobot_pi05
