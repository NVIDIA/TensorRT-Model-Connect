/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_repvgg/runtime/image_preprocess_seam.h"

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {
namespace {

stbir_filter resize_filter(const std::string& interpolation) {
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "bicubic")
        return STBIR_FILTER_CATMULLROM;
    throw std::invalid_argument("Unsupported timm RepVGG interpolation: " + interpolation);
}

void validate_config(const TimmRepvggPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("timm RepVGG input dimensions must be positive");
    if (config.crop_pct <= 0.0F || config.crop_pct > 1.0F)
        throw std::invalid_argument("timm RepVGG crop_pct must be in (0, 1]");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("timm RepVGG image mean/std must contain three channels");
    for (const float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("timm RepVGG image std must be non-zero");
    }
}

int32_t center_crop_offset(int32_t resized, int32_t target) {
    const int32_t difference = resized - target;
    const int32_t half = difference / 2;
    return (difference % 2 != 0 && half % 2 != 0) ? half + 1 : half;
}

} // namespace

TimmRepvggResizeShape compute_timm_repvgg_resize_shape(int32_t image_height, int32_t image_width,
                                                       const TimmRepvggPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("timm RepVGG source dimensions must be positive");
    validate_config(config);
    if (config.input_image_h == config.input_image_w) {
        const int32_t short_edge = static_cast<int32_t>(
            std::floor(static_cast<float>(config.input_image_h) / config.crop_pct));
        if (image_height <= image_width) {
            return {short_edge, static_cast<int32_t>(static_cast<int64_t>(short_edge) *
                                                     image_width / image_height)};
        }
        return {static_cast<int32_t>(static_cast<int64_t>(short_edge) * image_height / image_width),
                short_edge};
    }
    const float required = std::max(static_cast<float>(config.input_image_h) / image_height,
                                    static_cast<float>(config.input_image_w) / image_width) /
                           config.crop_pct;
    return {
        std::max(config.input_image_h, static_cast<int32_t>(std::floor(image_height * required))),
        std::max(config.input_image_w, static_cast<int32_t>(std::floor(image_width * required))),
    };
}

std::vector<float> preprocess_timm_repvgg_image(const float* image_pixels, int32_t image_height,
                                                int32_t image_width,
                                                const TimmRepvggPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("timm RepVGG source image must be non-empty");
    validate_config(config);
    const auto shape = compute_timm_repvgg_resize_shape(image_height, image_width, config);
    std::vector<float> resized(static_cast<std::size_t>(shape.height) * shape.width * 3U);
    if (stbir_resize(
            image_pixels, image_width, image_height,
            image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(), shape.width,
            shape.height, shape.width * 3 * static_cast<int32_t>(sizeof(float)), STBIR_RGB,
            STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP, resize_filter(config.interpolation)) == nullptr) {
        throw std::runtime_error("Failed to resize timm RepVGG input image");
    }
    const int32_t crop_y = center_crop_offset(shape.height, config.input_image_h);
    const int32_t crop_x = center_crop_offset(shape.width, config.input_image_w);
    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> output(3U * plane);
    for (int32_t y = 0; y < config.input_image_h; ++y) {
        for (int32_t x = 0; x < config.input_image_w; ++x) {
            const auto source =
                static_cast<std::size_t>(((crop_y + y) * shape.width + crop_x + x) * 3);
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto index = static_cast<std::size_t>(channel);
                output[index * plane + static_cast<std::size_t>(y) * config.input_image_w + x] =
                    (resized[source + index] - config.image_mean[index]) / config.image_std[index];
            }
        }
    }
    return output;
}

} // namespace trtmc
