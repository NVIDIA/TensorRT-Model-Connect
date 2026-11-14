/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/image.h"

#include <cstdint>
#include <vector>

namespace trtmc::gemma3n {

// Interleaved 8-bit RGB of one request image. UInt8 input is used as is;
// Float32 input in [0, 1] is quantized to the nearest 8-bit level, which
// recovers the exact pixels of any image decoded from 8-bit data.
struct Rgb8 {
    std::vector<uint8_t> pixels;
    int32_t height{0};
    int32_t width{0};
};
Rgb8 to_rgb8(const internal::ImageView& image);

// torch.nn.functional.interpolate(uint8 NCHW, mode="bilinear", antialias=True,
// align_corners=False) on CPU: width pass, then height pass, each in 16-bit
// fixed point and rounded to 8 bits; an axis whose size is unchanged is copied.
std::vector<uint8_t> resize_bilinear_aa(const std::vector<uint8_t>& rgb, int32_t height,
                                        int32_t width, int32_t out_height, int32_t out_width);

// The vision tower input: resized to size x size, rescaled by `rescale` in
// float32 and laid out as planar CHW.
std::vector<float> pixel_values(const Rgb8& image, int32_t size, float rescale);

} // namespace trtmc::gemma3n
