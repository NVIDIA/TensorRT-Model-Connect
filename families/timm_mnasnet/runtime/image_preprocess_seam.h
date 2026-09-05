/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmMnasnetPreprocessConfig {
    int32_t input_image_h;
    int32_t input_image_w;
    std::vector<float> image_mean;
    std::vector<float> image_std;
    float crop_pct;
    std::string interpolation;
};

struct TimmMnasnetResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmMnasnetResizeShape compute_timm_mnasnet_resize_shape(int32_t image_height, int32_t image_width,
                                                         const TimmMnasnetPreprocessConfig& config);

std::vector<float> preprocess_timm_mnasnet_image(const float* image_pixels, int32_t image_height,
                                                 int32_t image_width,
                                                 const TimmMnasnetPreprocessConfig& config);

} // namespace trtmc
