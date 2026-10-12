/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc {

struct DeformableDetrPreprocessConfig {
    std::int32_t input_image_h{800};
    std::int32_t input_image_w{800};
    // This checkpoint's preprocessor normalises, unlike RT-DETR's, which lists
    // the same statistics but disables them. Recorded in the bundle so the
    // runtime cannot drift from what the engine was built against.
    bool do_normalize{true};
    std::array<float, 3> image_mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> image_std{0.229F, 0.224F, 0.225F};
};

// Resize to the square the engine was built for, then normalise.
//
// The CLI delivers pixels already in [0, 1] (apps/cli/io.cpp divides by 255),
// so there is no second division here. Dividing again is the mistake that
// produced zero detections in RT-DETR.
std::vector<float> preprocess_deformable_detr_image(const float* pixels, std::int32_t image_height,
                                                    std::int32_t image_width,
                                                    const DeformableDetrPreprocessConfig& config);

} // namespace trtmc
