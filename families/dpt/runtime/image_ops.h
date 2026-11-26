/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::dpt {

struct PreprocessConfig {
    std::int32_t input_image_h{384};
    std::int32_t input_image_w{384};
    double rescale_factor{1.0 / 255.0};
    std::array<float, 3> image_mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> image_std{0.5F, 0.5F, 0.5F};
};

// What preprocess_image needs of a configuration: a positive engine input size,
// a finite positive rescale factor, and float32 means and standard deviations
// with which every 8-bit sample rescales and normalizes to a finite value,
// computed one operation at a time as preprocess_image computes it. Throws
// std::invalid_argument otherwise.
void check_preprocess_config(const PreprocessConfig& config);

// Interleaved RGB pixels in [0, 1] (8-bit samples divided by 255) back to the
// 8-bit samples the reference's PIL image holds: round(value * 255), clamped.
std::vector<std::uint8_t> to_rgb8(const float* pixels, std::int32_t height, std::int32_t width);

// Pillow 13.0.0 Image.resize(size, Image.BILINEAR) of an interleaved 8-bit RGB
// image: two separable passes with 22-bit fixed-point weights, each rounded
// to 8 bits, the vertical pass first when the height shrinks by more than
// twice as many pixels as the width. Adapted from Pillow; see NOTICE.
std::vector<std::uint8_t> resize_bilinear_rgb8(const std::uint8_t* rgb, std::int32_t height,
                                               std::int32_t width, std::int32_t out_height,
                                               std::int32_t out_width);

// The DPT image processor: resize to the engine input, rescale, normalize,
// planar CHW float32.
std::vector<float> preprocess_image(const float* pixels, std::int32_t height, std::int32_t width,
                                    const PreprocessConfig& config);

// torch.nn.functional.interpolate(mode="bicubic", align_corners=False) of one
// row-major [height, width] float map: cubic convolution with A = -0.75 and
// border-clamped taps. Adapted from PyTorch; see NOTICE.
std::vector<float> resize_bicubic(const float* source, std::int32_t source_h, std::int32_t source_w,
                                  std::int32_t target_h, std::int32_t target_w);

} // namespace trtmc::dpt
