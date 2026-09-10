/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmRes2NetPreprocessConfig {
    int32_t input_image_h{224};
    int32_t input_image_w{224};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
    float crop_pct{0.9F};
    std::string interpolation{"bicubic"};
};

struct TimmRes2NetResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmRes2NetResizeShape compute_timm_res2net_resize_shape(int32_t image_height, int32_t image_width,
                                                         const TimmRes2NetPreprocessConfig& config);

std::vector<float> preprocess_timm_res2net_image(const float* image_pixels, int32_t image_height,
                                                 int32_t image_width,
                                                 const TimmRes2NetPreprocessConfig& config);

} // namespace trtmc
