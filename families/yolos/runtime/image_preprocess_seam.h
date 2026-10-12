/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct YolosPreprocessConfig {
    int32_t input_image_h{512};
    int32_t input_image_w{864};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// Resize straight to the engine's fixed input size, then normalize into CHW.
// YOLOS position embeddings are not interpolated by this family, so the engine
// only ever accepts the checkpoint's native resolution and the aspect ratio of
// the source image is not preserved.
std::vector<float> preprocess_yolos_image(const float* image_pixels, int32_t image_height,
                                          int32_t image_width, const YolosPreprocessConfig& config);

} // namespace trtmc
