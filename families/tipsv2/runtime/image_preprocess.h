/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::tipsv2 {

struct PreprocessConfig {
    std::int32_t input_image_h{448};
    std::int32_t input_image_w{448};
    // The reference divides the resized 8-bit image by 255 and stops there. There is no ImageNet
    // mean or deviation: TIPSv2 is trained on pixels in [0, 1], and subtracting a mean here would
    // move every pixel and nothing loudly.
    float rescale_factor{1.0F / 255.0F};
};

// Squash the whole image onto the square grid - no aspect-ratio rule and no centre crop - with
// Pillow's bilinear filter, then rescale. Returns CHW float32 of
// `3 * input_image_h * input_image_w`. Exposed so the unit test can pin the rule without an
// engine.
std::vector<float> preprocess_tipsv2_image(const float* image_pixels, std::int32_t image_height,
                                           std::int32_t image_width,
                                           const PreprocessConfig& config);

} // namespace trtmc::tipsv2
