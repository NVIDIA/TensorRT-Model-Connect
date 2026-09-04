/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmVggPreprocessConfig {
    int32_t input_image_h;
    int32_t input_image_w;
    std::vector<float> image_mean;
    std::vector<float> image_std;
    float crop_pct;
    std::string interpolation;
};

struct TimmVggResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmVggResizeShape compute_timm_vgg_resize_shape(int32_t image_height, int32_t image_width,
                                                 const TimmVggPreprocessConfig& config);

std::vector<float> preprocess_timm_vgg_image(const float* image_pixels, int32_t image_height,
                                             int32_t image_width,
                                             const TimmVggPreprocessConfig& config);

} // namespace trtmc
