/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TableTransformerPreprocessConfig {
    int32_t input_image_h{1000};
    int32_t input_image_w{1000};
    int32_t shortest_edge{800};
    int32_t longest_edge{1000};
    // New bundles accept the actual resized shape; old bundles require an exact canvas.
    bool dynamic_image_shapes{false};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

struct TableTransformerResizeShape {
    int32_t height{0};
    int32_t width{0};
};

TableTransformerResizeShape
compute_table_transformer_resize_shape(int32_t image_height, int32_t image_width,
                                       const TableTransformerPreprocessConfig& config);

std::vector<float>
preprocess_table_transformer_image(const float* image_pixels, int32_t image_height,
                                   int32_t image_width,
                                   const TableTransformerPreprocessConfig& config);

} // namespace trtmc
