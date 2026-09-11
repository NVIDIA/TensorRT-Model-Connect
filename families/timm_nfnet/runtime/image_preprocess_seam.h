/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TimmNfnetPreprocessConfig {
    int32_t input_image_h{224};
    int32_t input_image_w{224};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
    float crop_pct{0.9F};
    std::string interpolation{"bicubic"};
    // timm's crop_mode. "center" keeps the aspect ratio and crops; "squash"
    // resizes both axes to the same length first, which every dm_nfnet
    // checkpoint asks for.
    std::string crop_mode{"center"};
};

struct TimmNfnetResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TimmNfnetResizeShape compute_timm_nfnet_resize_shape(int32_t image_height, int32_t image_width,
                                                     const TimmNfnetPreprocessConfig& config);

std::vector<float> preprocess_timm_nfnet_image(const float* image_pixels, int32_t image_height,
                                               int32_t image_width,
                                               const TimmNfnetPreprocessConfig& config);

} // namespace trtmc
