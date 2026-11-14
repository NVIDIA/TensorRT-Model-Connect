/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Adapted from transformers, Pillow and PyTorch; see families/siglip2/NOTICE.

#include <cstdint>
#include <utility>
#include <vector>

namespace trtmc {

struct Siglip2PreprocessConfig {
    std::int32_t patch_size{16};
    std::int32_t max_num_patches{256};
    // Side of the square position table the vision engine carries.
    std::int32_t position_grid{16};
    double rescale_factor{1.0 / 255.0};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
};

// One image cut into its NaFlex grid of patches.
struct Siglip2Patches {
    std::int32_t grid_height{0};
    std::int32_t grid_width{0};
    // [grid_height * grid_width, patch * patch * 3], patches in row-major grid order and
    // each patch in (row, column, channel) order: the layout the patch Linear expects.
    std::vector<float> pixel_values;
    // [grid_height * grid_width, position_grid^2]: the antialiased bilinear taps that
    // resample the square position table to this grid.
    std::vector<float> position_weights;
};

// The reference processor's binary search: the largest aspect-preserving size whose
// sides are whole patches and whose patch count fits the budget. Returns (height, width)
// in pixels.
std::pair<std::int32_t, std::int32_t> siglip2_target_size(std::int32_t height, std::int32_t width,
                                                          std::int32_t patch_size,
                                                          std::int32_t max_num_patches);

// Pillow 13's 8-bit bilinear resize of interleaved RGB, reproduced coefficient for
// coefficient (fixed point, with Pillow 13's choice of which pass runs first).
std::vector<std::uint8_t> siglip2_resize_bilinear(const std::uint8_t* rgb, std::int32_t height,
                                                  std::int32_t width, std::int32_t out_height,
                                                  std::int32_t out_width);

// The taps of torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
// antialias=True) along one axis, as a dense [output, input] matrix.
std::vector<double> siglip2_axis_taps(std::int32_t input, std::int32_t output);

// Float RGB in [0, 1], interleaved [height, width, 3], to the vision engine's inputs.
Siglip2Patches preprocess_siglip2_image(const float* rgb, std::int32_t height, std::int32_t width,
                                        const Siglip2PreprocessConfig& config);

} // namespace trtmc
