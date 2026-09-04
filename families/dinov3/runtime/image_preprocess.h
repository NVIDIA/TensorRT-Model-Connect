/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

// Hugging Face DINOv3ViTImageProcessor defaults. The public pipeline accepts
// already-rescaled RGB HWC floats, so the remaining transform is direct
// bilinear resize followed by ImageNet normalization and NCHW packing.
struct Dinov3PreprocessConfig {
    int32_t input_image_h{224};
    int32_t input_image_w{224};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
    std::string interpolation{"bilinear"};
    bool do_center_crop{false};
    float crop_pct{1.0F};
};

std::vector<float> preprocess_dinov3_image(const float* image_pixels, int32_t image_height,
                                           int32_t image_width,
                                           const Dinov3PreprocessConfig& config);

} // namespace trtmc
