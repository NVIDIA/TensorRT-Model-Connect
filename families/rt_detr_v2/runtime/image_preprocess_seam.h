/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// RT-DETR v2 resizes to a fixed square and does NOT normalise.
//
// The checkpoint's preprocessor_config.json still lists image_mean and
// image_std, but it sets do_normalize to false, so those values are inert. A
// builder that reads them - which is the right thing to do for every
// classifier family in this repository - produces an input with mean -0.12 over
// [-2.10, 2.64] instead of mean 0.42 over [0, 1], and the model still runs.
struct RtDetrPreprocessConfig {
    int32_t input_image_h{640};
    int32_t input_image_w{640};
    // Aspect ratio is deliberately not preserved: a 640x382 source becomes
    // 640x640, and the boxes are mapped back using the original size.
    bool preserve_aspect_ratio{false};
};

// Bilinear resize to the configured square. 'pixels' is HWC and ALREADY in
// [0, 1]: apps/cli/io.cpp divides by 255 when it decodes the image, so a second
// division here would darken the input by 255x and the detector would return
// nothing at all. The result is CHW, still in [0, 1].
std::vector<float> preprocess_rt_detr_image(const float* pixels, int32_t image_height,
                                            int32_t image_width,
                                            const RtDetrPreprocessConfig& config);

} // namespace trtmc
