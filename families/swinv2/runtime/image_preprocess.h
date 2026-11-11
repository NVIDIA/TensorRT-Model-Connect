/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::swinv2 {

struct PreprocessConfig {
    std::int32_t input_image_h{256};
    std::int32_t input_image_w{256};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// The checkpoint's processor squashes the whole image onto the square with
// Pillow's bicubic filter: no aspect-ratio preservation and no centre crop.
// Returns planar float32 RGB, normalized. Exposed so the unit test can pin the
// framing and the filter without running an engine.
std::vector<float> preprocess_image(const float* image_pixels, std::int32_t image_height,
                                    std::int32_t image_width, const PreprocessConfig& config);

} // namespace trtmc::swinv2
