/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmXceptionPreprocessConfig {
    int32_t input_image_h{224};
    int32_t input_image_w{224};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
    float crop_pct{0.9F};
    std::string interpolation{"bicubic"};
};

struct TimmXceptionResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmXceptionResizeShape
compute_timm_xception_resize_shape(int32_t image_height, int32_t image_width,
                                   const TimmXceptionPreprocessConfig& config);

std::vector<float> preprocess_timm_xception_image(const float* image_pixels, int32_t image_height,
                                                  int32_t image_width,
                                                  const TimmXceptionPreprocessConfig& config);

} // namespace trtmc
