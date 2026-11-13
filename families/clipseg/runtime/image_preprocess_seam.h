/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct ClipsegPreprocessConfig {
    std::int32_t input_image_h{352};
    std::int32_t input_image_w{352};
    std::vector<float> image_mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::vector<float> image_std{0.26862954F, 0.26130258F, 0.27577711F};
};

struct ClipsegResizeShape {
    std::int32_t height{0};
    std::int32_t width{0};
};

// Direct resize to the checkpoint dimensions, preserving the full source image.
ClipsegResizeShape compute_clipseg_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                                const ClipsegPreprocessConfig& config);

std::vector<float> preprocess_clipseg_image(const float* image_pixels, std::int32_t image_height,
                                            std::int32_t image_width,
                                            const ClipsegPreprocessConfig& config);

} // namespace trtmc
