/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Dinov2PreprocessConfig {
    std::int32_t input_image_h{224};
    std::int32_t input_image_w{224};
    std::int32_t shortest_edge{256};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

struct Dinov2ResizeShape {
    std::int32_t height{0};
    std::int32_t width{0};
};

// Maps a source pixel onto the processed crop:
//   processed_x = scale_x * source_x + offset_x
// The task contract reports each patch's footprint in ORIGINAL-image
// coordinates, so the runtime has to invert this. Returning it from the seam
// keeps the resize rule and its inverse in one place instead of letting the
// pipeline re-derive a second, silently different one.
struct Dinov2SourceTransform {
    double scale_x{1.0};
    double scale_y{1.0};
    double offset_x{0.0};
    double offset_y{0.0};
};

// Shortest edge to shortest_edge, the other edge floored. Exposed so the unit
// test can pin the rounding rule without running an engine.
Dinov2ResizeShape compute_dinov2_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                              const Dinov2PreprocessConfig& config);

Dinov2SourceTransform compute_dinov2_source_transform(std::int32_t image_height,
                                                      std::int32_t image_width,
                                                      const Dinov2PreprocessConfig& config);

std::vector<float> preprocess_dinov2_image(const float* image_pixels, std::int32_t image_height,
                                           std::int32_t image_width,
                                           const Dinov2PreprocessConfig& config);

} // namespace trtmc
