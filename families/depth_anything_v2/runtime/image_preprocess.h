/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Hugging Face DPTImageProcessor defaults, as used by Depth Anything V2. The
// public pipeline accepts already-rescaled RGB HWC floats, so the remaining
// transform is direct bilinear resize to the engine's one fixed square
// resolution, followed by ImageNet normalization and NCHW packing. Unlike
// the reference processor this does not preserve aspect ratio: the engine
// this feeds was built for exactly one square input.
struct DepthAnythingV2PreprocessConfig {
    int32_t input_image_size{518};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

std::vector<float>
preprocess_depth_anything_v2_image(const float* image_pixels, int32_t image_height,
                                   int32_t image_width,
                                   const DepthAnythingV2PreprocessConfig& config);

} // namespace trtmc
