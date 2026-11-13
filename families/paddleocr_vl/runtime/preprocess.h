/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

std::vector<float> prepare_paddleocr_image(const float* pixels, int32_t height, int32_t width,
    int32_t profile_height, int32_t profile_width, int32_t min_pixels, int32_t max_pixels);
std::vector<int32_t> paddleocr_position_ids(const std::vector<int32_t>& ids, int32_t image_id,
    int32_t image_sequence, int32_t grid_height, int32_t grid_width, int32_t profile_length);

struct PaddleOcrPreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    std::vector<float> image_mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::vector<float> image_std{0.26862954F, 0.26130258F, 0.27577711F};
};

struct PaddleOcrResizeShape {
    std::int32_t height{0};
    std::int32_t width{0};
};

// Direct resize to the checkpoint dimensions, preserving the full source image.
PaddleOcrResizeShape compute_paddleocr_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                          const PaddleOcrPreprocessConfig& config);

std::vector<float> preprocess_paddleocr_image(const float* image_pixels, std::int32_t image_height,
                                         std::int32_t image_width,
                                         const PaddleOcrPreprocessConfig& config);

} // namespace trtmc
