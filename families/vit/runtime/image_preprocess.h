/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Pillow's filter numbers, as a preprocessor config spells them.
enum class VitResample : std::int32_t { kBilinear = 2, kBicubic = 3 };

struct VitPreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    VitResample resample{VitResample::kBilinear};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
};

// ViT squashes the whole image onto the square: no aspect-ratio preservation
// and no centre crop. Exposed so the unit test can pin both the framing and
// the choice of filter without running an engine.
std::vector<float> preprocess_vit_image(const float* image_pixels, std::int32_t image_height,
                                        std::int32_t image_width,
                                        const VitPreprocessConfig& config);

} // namespace trtmc
