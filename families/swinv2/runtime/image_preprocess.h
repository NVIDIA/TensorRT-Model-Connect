/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::swinv2 {

struct PreprocessConfig {
    std::int32_t input_image_h{256};
    std::int32_t input_image_w{256};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// The checkpoint's processor squashes the whole image onto the square with
// Pillow's bicubic filter: no aspect-ratio preservation and no centre crop.
// Returns planar float32 RGB, normalized. Exposed so the unit test can pin the
// framing and the filter without running an engine.
std::vector<float> preprocess_image(const float* image_pixels, std::int32_t image_height,
                                    std::int32_t image_width, const PreprocessConfig& config);

namespace detail {

// Internal seams used by the focused tests: the resampler's index arithmetic,
// which leaves the int32 range at image sizes far too large to allocate in a
// test. Production callers use preprocess_image().

struct TapRange {
    std::int32_t first{0};
    std::int32_t end{0};
};

// The input taps [first, end) of the output sample centred at `center`, with
// Pillow's rounding, clamped to the input.
TapRange pillow_tap_range(double center, double support, std::int32_t input_size);

// Byte offset of input tap `tap` along an axis whose taps lie `pixels_per_tap`
// RGB pixels apart: 1 along a row, the image width along a column.
std::size_t pillow_tap_byte_offset(std::int32_t tap, std::int32_t pixels_per_tap);

// Pillow 13's pass order: true when the vertical pass runs first.
bool pillow_vertical_pass_first(std::int32_t input_h, std::int32_t input_w, std::int32_t output_h,
                                std::int32_t output_w);

} // namespace detail

} // namespace trtmc::swinv2
