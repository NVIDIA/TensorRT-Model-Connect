/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Blip2PreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    std::vector<float> image_mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::vector<float> image_std{0.26862954F, 0.26130258F, 0.27577711F};
};

struct Blip2ResizeShape {
    std::int32_t height{0};
    std::int32_t width{0};
};

// Direct resize to the checkpoint dimensions, preserving the full source image.
Blip2ResizeShape compute_blip2_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                            const Blip2PreprocessConfig& config);

std::vector<float> preprocess_blip2_image(const float* image_pixels, std::int32_t image_height,
                                          std::int32_t image_width,
                                          const Blip2PreprocessConfig& config);

} // namespace trtmc
