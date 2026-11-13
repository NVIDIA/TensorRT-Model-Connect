/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <vector>

namespace trtmc::lerobot_pi05 {

enum class NormalizationMode { kIdentity, kMeanStd, kRange };

// Per-feature normalization in LeRobot's NormalizerProcessorStep arithmetic, in float32 and in
// the same operation order. Empty statistics mean the value passes through, as LeRobot skips a
// feature that has none.
//   kMeanStd: offset = mean, scale = std;  (x - mean) / (std + eps)  and  x * std + mean
//   kRange:   QUANTILES, QUANTILE10 and MIN_MAX. offset = q01, q10 or min; scale = the spread to
//             q99, q90 or max, eps where it is zero; 2 (x - offset) / scale - 1 and
//             (x + 1) * scale / 2 + offset. LeRobot writes the min/max inverse as
//             (x + 1) / 2 * scale, which is the same float32 value because halving is exact.
struct Normalization {
    NormalizationMode mode{NormalizationMode::kIdentity};
    float eps{1.0e-8F};
    std::vector<float> offset;
    std::vector<float> scale;
    bool identity() const { return mode == NormalizationMode::kIdentity || offset.empty(); }
};

float normalize_value(const Normalization& normalization, std::size_t index, float value);
float unnormalize_value(const Normalization& normalization, std::size_t index, float value);

} // namespace trtmc::lerobot_pi05
