/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::glm_ocr {

// The checkpoint's Glm46VImageProcessor (torchvision backend) and the vision rotary table.
struct ImageConfig {
    int patch_size{0};
    int temporal_patch_size{0};
    int merge_size{0};
    std::int64_t min_pixels{0};
    std::int64_t max_pixels{0};
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
    int head_dim{0};
    float rope_theta{0.0F};
    int max_patches{0};
};

struct ImageSize {
    int height{0};
    int width{0};
};

// smart_resize of the processor: a multiple of patch_size * merge_size on both sides whose
// two-frame pixel count lies in [min_pixels, max_pixels]. Python's float arithmetic and
// round-half-even are reproduced exactly.
ImageSize smart_resize(int height, int width, const ImageConfig& config);

// Host RGB image (UInt8, or Float32 in [0, 1] with values on the 1/255 grid) as RGB8 HWC.
std::vector<std::uint8_t> to_rgb8(const internal::ImageView& image);

// torchvision's antialiased bicubic resize of uint8 images on the CPU: PyTorch's int16
// fixed-point kernel, horizontal pass then vertical pass, each skipped when its size is kept
// (adapted code; see the family NOTICE).
std::vector<std::uint8_t> resize_bicubic(const std::vector<std::uint8_t>& rgb, int height,
                                         int width, int out_height, int out_width);

// One image ready for the vision engine.
struct VisionInput {
    int grid_h{0}; // patch rows
    int grid_w{0}; // patch columns
    // [patches, channels * temporal_patch_size * patch_size^2] in merge-window order.
    std::vector<float> patches;
    // [patches, head_dim / 2]: row frequencies, then column frequencies.
    std::vector<float> rotary_cos;
    std::vector<float> rotary_sin;
    int num_patches() const { return grid_h * grid_w; }
};

VisionInput preprocess_image(const internal::ImageView& image, const ImageConfig& config);

} // namespace trtmc::glm_ocr
