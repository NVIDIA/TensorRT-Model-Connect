/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::hunyuan_vl {
struct ImageSettings {
    int patch{}, merge{};
    int64_t min_pixels{}, max_pixels{};
    std::array<float, 3> mean{}, std{};
    double rescale_factor{};
};
// One image resized by the reference's smart_resize; rows/cols count patches.
struct ImagePlan {
    internal::ImageView image;
    int height{}, width{}, patch{}, merge{};
    int rows() const { return height / patch; }
    int cols() const { return width / patch; }
    int patches() const { return rows() * cols(); }
    int grid_rows() const { return rows() / merge; }
    int grid_cols() const { return cols() / merge; }
    // Placeholder tokens: one per merged cell, one row separator per merged row, begin and end.
    int tokens() const { return grid_rows() * (grid_cols() + 1) + 2; }
};
ImagePlan plan_image(internal::ImageView image, const ImageSettings& settings, int max_patches);
// Normalized pixels [patches, 3 * patch * patch] in 2x2 merge-block order, channel-major per patch.
std::vector<float> patchify(const ImagePlan& plan, const ImageSettings& settings);
// Bilinear (align_corners=false) corners of the learned edge x edge position grid, in the same
// patch order: ids hold table rows (the class-token row 0 is skipped) and lambdas h0, h1, w0, w1.
void position_corners(const ImagePlan& plan, int edge, std::vector<int32_t>& ids,
                      std::vector<float>& lambdas);
// Resampling exposed for the family's preprocessing parity test.
std::vector<unsigned char> resize_bicubic_antialias_u8(const unsigned char* raw, int32_t width,
                                                       int32_t height, int32_t target_width,
                                                       int32_t target_height);
} // namespace trtmc::hunyuan_vl
