/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct DepthAnythingPreprocessConfig {
    int32_t input_image_h{518};
    int32_t input_image_w{518};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// Resize straight to the engine's fixed square input, then normalize into CHW.
// This family does not interpolate the backbone position embeddings, so the
// engine only accepts the checkpoint's native resolution and the source aspect
// ratio is not preserved.
std::vector<float> preprocess_depth_anything_image(const float* image_pixels, int32_t image_height,
                                                   int32_t image_width,
                                                   const DepthAnythingPreprocessConfig& config);

} // namespace trtmc
