/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::moge {

struct GeometryResult {
    std::vector<float> points;         // [H, W, 3], metric OpenCV camera coordinates
    std::vector<float> depth;          // [H, W], meters
    std::vector<uint8_t> mask;         // [H, W], values are 0 or 1
    std::array<float, 9> intrinsics{}; // normalized row-major [3, 3]
    int32_t height{0};
    int32_t width{0};
};

class IGeometryEstimator {
  public:
    virtual ~IGeometryEstimator() = default;
    virtual GeometryResult estimate_geometry(const float* pixels, int32_t height,
                                             int32_t width) = 0;
};

} // namespace trtmc::moge
