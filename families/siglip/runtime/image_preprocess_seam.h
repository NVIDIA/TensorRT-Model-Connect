/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct SiglipPreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
};

// SigLIP squashes the whole image onto the square grid: no aspect-ratio
// preservation and no centre crop, which is the opposite of CLIP's rule and
// the reason this seam is not shared with it. Exposed so the unit test can
// pin the rule without running an engine.
std::vector<float> preprocess_siglip_image(const float* image_pixels, std::int32_t image_height,
                                           std::int32_t image_width,
                                           const SiglipPreprocessConfig& config);

} // namespace trtmc
