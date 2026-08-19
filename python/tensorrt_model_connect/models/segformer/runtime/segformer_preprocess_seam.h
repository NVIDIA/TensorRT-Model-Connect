/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "decoded_image.h"

#include <cstdint>
#include <vector>

namespace trtmc {

struct SegformerPreprocessConfig {
    int32_t num_classes{150};
    int32_t input_image_h{512};
    int32_t input_image_w{512};
    int32_t output_h{128};
    int32_t output_w{128};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

std::vector<float> preprocess_segformer_image(const float* image_pixels, int32_t image_height,
                                              int32_t image_width,
                                              const SegformerPreprocessConfig& config);

std::vector<float> preprocess_segformer_image(const runtime::adapters::io::DecodedImage& image,
                                              const SegformerPreprocessConfig& config);

} // namespace trtmc
