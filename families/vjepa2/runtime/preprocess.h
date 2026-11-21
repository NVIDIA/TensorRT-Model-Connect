/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <vector>
namespace trtmc::vjepa2 {
struct ImageGeometry {
    int32_t height, width, top, left;
};
ImageGeometry image_geometry(int32_t height, int32_t width, int32_t shortest, int32_t crop);
// Contiguous unit RGB float32 HWC -> antialiased resize, center crop, CHW.
std::vector<float> preprocess(const float* rgb, int32_t height, int32_t width, int32_t shortest,
                              int32_t crop);
} // namespace trtmc::vjepa2
