/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmMobilenetv3PreprocessConfig {
    int32_t input_image_h;
    int32_t input_image_w;
    std::vector<float> image_mean;
    std::vector<float> image_std;
    float crop_pct;
    std::string interpolation;
};

struct TimmMobilenetv3ResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmMobilenetv3ResizeShape
compute_timm_mobilenetv3_resize_shape(int32_t image_height, int32_t image_width,
                                      const TimmMobilenetv3PreprocessConfig& config);

std::vector<float> preprocess_timm_mobilenetv3_image(const float* image_pixels,
                                                     int32_t image_height, int32_t image_width,
                                                     const TimmMobilenetv3PreprocessConfig& config);

} // namespace trtmc
