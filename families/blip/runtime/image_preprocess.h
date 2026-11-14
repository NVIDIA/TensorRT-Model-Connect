/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::blip {

// Pillow's filter numbers, as a preprocessor config spells them.
enum class BlipResample : std::int32_t { kBilinear = 2, kBicubic = 3 };

struct BlipPreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    BlipResample resample{BlipResample::kBicubic};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
};

// ViT squashes the whole image onto the square: no aspect-ratio preservation
// and no centre crop. Exposed so the unit test can pin both the framing and
// the choice of filter without running an engine.
std::vector<float> preprocess_blip_image(const float* image_pixels, std::int32_t image_height,
                                         std::int32_t image_width,
                                         const BlipPreprocessConfig& config);

} // namespace trtmc::blip
