/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct ClipPreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    std::vector<float> image_mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::vector<float> image_std{0.26862954F, 0.26130258F, 0.27577711F};
};

struct ClipResizeShape {
    std::int32_t height{0};
    std::int32_t width{0};
};

// Shortest edge to the target, the other edge floored. Exposed so the unit
// test can pin the rounding rule without running an engine.
ClipResizeShape compute_clip_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                          const ClipPreprocessConfig& config);

std::vector<float> preprocess_clip_image(const float* image_pixels, std::int32_t image_height,
                                         std::int32_t image_width,
                                         const ClipPreprocessConfig& config);

} // namespace trtmc
