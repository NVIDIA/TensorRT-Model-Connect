/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::trocr {

// The checkpoint's ViTImageProcessor: squash to height x width with Pillow's
// bilinear filter on 8-bit RGB, rescale, normalize, channel-first.
struct PreprocessConfig {
    std::int32_t height{384};
    std::int32_t width{384};
    double rescale_factor{1.0 / 255.0};
    std::array<float, 3> image_mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> image_std{0.5F, 0.5F, 0.5F};
};

// Interleaved 8-bit RGB as Pillow's convert("RGB") gives it: grayscale is
// replicated, an alpha channel is dropped. Float32 values in [0, 1] are
// rounded to the nearest of the 256 levels the reference processor works on.
std::vector<std::uint8_t> to_rgb8(const internal::ImageView& image);

// Pillow's two-pass resampling of interleaved 8-bit RGB with its 22-bit
// fixed-point coefficients and rounding, in Pillow 13's pass order.
std::vector<std::uint8_t> resize_bilinear(const std::uint8_t* rgb, std::int32_t height,
                                          std::int32_t width, std::int32_t output_height,
                                          std::int32_t output_width);

// pixel_values [3, height, width]: float32(u8 * rescale_factor in double),
// then (value - mean) / std in float32, exactly as the reference computes it.
std::vector<float> preprocess(const internal::ImageView& image, const PreprocessConfig& config);

} // namespace trtmc::trocr
