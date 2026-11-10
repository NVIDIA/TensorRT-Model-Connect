/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::minimax_h3 {

struct ReferenceImageShape {
    int32_t height;
    int32_t width;
};

// Released original H3 image-reference setup: aspect <= 4, short edge 2048,
// ties-to-even rounding to 32 pixels, including upscaling, without an area cap.
ReferenceImageShape reference_image_shape(int32_t height, int32_t width);

// RGB uint8, full image box, Pillow 12.2 LANCZOS semantics. Horizontal and
// vertical passes each round/clamp to uint8. The intermediate is a rolling
// row window, not a second full-size image. No crop or pre-reduction.
std::vector<uint8_t> resize_reference_rgb_lanczos(const uint8_t* rgb, std::size_t bytes,
                                                  int32_t height, int32_t width,
                                                  ReferenceImageShape target);

struct PreparedReferenceImage {
    ReferenceImageShape shape;
    std::vector<uint8_t> rgb;
};

PreparedReferenceImage prepare_reference_image(const uint8_t* rgb, std::size_t bytes,
                                               int32_t height, int32_t width);

// VisualVAE input is FP32 [1,3,1,tile_height,tile_width], with the checkpoint's
// ImageNet normalization, NOT Qwen's fused [-1,1] normalization. An image
// remains one frame; temporal duplication belongs only to Qwen patch packing.
std::vector<float> pack_visual_reference_rgb_tile(const PreparedReferenceImage& image, int32_t top,
                                                  int32_t left, int32_t tile_height,
                                                  int32_t tile_width);

} // namespace trtmc::minimax_h3
