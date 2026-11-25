/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::mask2former {

struct PreprocessConfig {
    std::int32_t height{0};
    std::int32_t width{0};
    double rescale_factor{0.0};
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
};

// Image.resize((width, height), Image.BILINEAR) of Pillow 13 on an 8-bit RGB image: separable
// fixed-point passes, each rounded to 8 bits, in Pillow's pass order. Input and output are
// contiguous HWC.
std::vector<std::uint8_t> resize_bilinear_rgb8(const std::uint8_t* pixels, std::int32_t height,
                                               std::int32_t width, std::int32_t out_height,
                                               std::int32_t out_width);

// True when every 8-bit value normalizes to a finite float32 the way preprocess() computes it: a
// positive rescale factor that keeps 255 within float range, and no mean or std that makes
// (value - mean) / std overflow.
bool normalization_is_finite(const PreprocessConfig& config);

// Mask2FormerImageProcessor on an 8-bit RGB image: the resize above, then float32(value *
// rescale_factor) and (value - mean) / std per channel. Returns CHW float32 pixel values; throws
// std::invalid_argument when the normalization is not finite.
std::vector<float> preprocess(const std::uint8_t* pixels, std::int32_t height, std::int32_t width,
                              const PreprocessConfig& config);

// Class map of CHW scores resized bilinearly (torch interpolate, align_corners=False) to height x
// width: the first class with the largest resized score at each pixel.
std::vector<std::int32_t> resize_argmax(const float* scores, std::int32_t classes,
                                        std::int32_t score_height, std::int32_t score_width,
                                        std::int32_t height, std::int32_t width);

} // namespace trtmc::mask2former
