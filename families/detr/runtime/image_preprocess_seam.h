/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct DetrPreprocessConfig {
    int32_t input_image_h{800};
    int32_t input_image_w{800};
    int32_t shortest_edge{800};
    int32_t longest_edge{1333};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

struct DetrResizeShape {
    int32_t height{0};
    int32_t width{0};
};

DetrResizeShape compute_detr_resize_shape(int32_t image_height, int32_t image_width,
                                          const DetrPreprocessConfig& config);

std::vector<float> preprocess_detr_image(const float* image_pixels, int32_t image_height,
                                         int32_t image_width, const DetrPreprocessConfig& config);

} // namespace trtmc
