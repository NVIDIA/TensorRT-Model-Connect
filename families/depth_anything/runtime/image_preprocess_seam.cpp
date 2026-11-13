/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything/runtime/image_preprocess_seam.h"

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <cstddef>
#include <stdexcept>

namespace trtmc {
namespace {

void validate(const DepthAnythingPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("depth_anything engine input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("depth_anything image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("depth_anything image std must be non-zero");
    }
}

} // namespace

std::vector<float> preprocess_depth_anything_image(const float* image_pixels, int32_t image_height,
                                                   int32_t image_width,
                                                   const DepthAnythingPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("depth_anything source image must be non-empty");
    validate(config);

    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;
    std::vector<float> resized(static_cast<std::size_t>(out_h) * out_w * 3U);
    if (stbir_resize(image_pixels, image_width, image_height,
                     image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(), out_w,
                     out_h, out_w * 3 * static_cast<int32_t>(sizeof(float)), STBIR_RGB,
                     STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE) == nullptr) {
        throw std::runtime_error("Failed to resize depth_anything input image");
    }

    const auto plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::size_t index = 0; index < plane; ++index) {
        const std::size_t source = index * 3U;
        pixel_values[index] = (resized[source] - config.image_mean[0]) / config.image_std[0];
        pixel_values[plane + index] =
            (resized[source + 1] - config.image_mean[1]) / config.image_std[1];
        pixel_values[2U * plane + index] =
            (resized[source + 2] - config.image_mean[2]) / config.image_std[2];
    }
    return pixel_values;
}

} // namespace trtmc
