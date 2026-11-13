/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// BiRefNet resizes to a fixed square and DOES normalise, unlike rt_detr_v2.
// The statistics come from the checkpoint's own handler, not from a
// preprocessor config: the repository ships none.
struct BiRefNetPreprocessConfig {
    int32_t input_image_h{1024};
    int32_t input_image_w{1024};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// 'pixels' is HWC and already in [0, 1]: apps/cli/io.cpp divides by 255 when it
// decodes. The result is CHW, resized and normalised.
std::vector<float> preprocess_birefnet_image(const float* pixels, int32_t image_height,
                                             int32_t image_width,
                                             const BiRefNetPreprocessConfig& config);

// Threshold the logit map into per-pixel class ids, resampling back to the
// source resolution with nearest neighbour so no new class values appear.
std::vector<int32_t> mask_from_logits(const float* logits, int32_t logit_height,
                                      int32_t logit_width, int32_t out_height, int32_t out_width,
                                      float threshold);

} // namespace trtmc
