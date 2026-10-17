/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Transformers BitImageProcessor as configured by DINOv2 checkpoints: resize
// the shortest edge with Pillow bicubic, center-crop, rescale and normalize.
struct Dinov2PreprocessConfig {
    int32_t input_image_h{224};
    int32_t input_image_w{224};
    int32_t resize_shortest_edge{256};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

// Where the engine input came from: the resized image and the crop origin in it.
struct Dinov2ImageGeometry {
    int32_t resized_h{0};
    int32_t resized_w{0};
    int32_t crop_y{0};
    int32_t crop_x{0};
};

Dinov2ImageGeometry compute_dinov2_image_geometry(int32_t image_height, int32_t image_width,
                                                  const Dinov2PreprocessConfig& config);

// `rgb` is contiguous HWC uint8 RGB. Returns NCHW float32 pixel values.
std::vector<float> preprocess_dinov2_image(const uint8_t* rgb, int32_t image_height,
                                           int32_t image_width,
                                           const Dinov2PreprocessConfig& config);

} // namespace trtmc
