/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::paligemma {

struct ImageNormalization {
    std::int32_t size{224};
    double rescale_factor{1.0 / 255.0};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> std{0.5F, 0.5F, 0.5F};
};

// Pillow's bicubic resize of 8-bit interleaved RGB, reproduced coefficient for
// coefficient (two separable passes, 22-bit fixed point, 8-bit rounding after
// each pass): the reference processor resamples with PIL (resample=3).
std::vector<std::uint8_t> resize_pillow_bicubic(const std::uint8_t* rgb, std::int32_t height,
                                                std::int32_t width, std::int32_t output_height,
                                                std::int32_t output_width);

// The reference SigLIP processing: squash the whole image onto the square (no
// aspect-ratio preservation, no crop), rescale in float64 and store float32,
// then normalize in float32. Returns planar CHW float32.
std::vector<float> preprocess_image(const std::uint8_t* rgb, std::int32_t height,
                                    std::int32_t width, const ImageNormalization& config);

} // namespace trtmc::paligemma
