/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct DetrPreprocessConfig {
    std::int32_t input_image_h{800};
    std::int32_t input_image_w{800};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// `pixels` is an interleaved RGB image in [0, 1], the layout the CLI's
// image reader produces. The result is planar CHW for the engine.
//
// DETR predicts boxes normalised to its own input, so a plain resize keeps the
// mapping back to the source image a single multiply per axis. Padding would
// need the position embedding to know which pixels are real.
std::vector<float> preprocess_detr_image(const float* pixels, std::int32_t height,
                                         std::int32_t width, const DetrPreprocessConfig& config);

} // namespace trtmc
