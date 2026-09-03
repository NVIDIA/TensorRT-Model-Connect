/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/timm_repvgg/image_preprocess_seam.h"

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

stbir_filter resolve_timm_repvgg_resize_filter(const std::string& interpolation) {
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "bicubic")
        return STBIR_FILTER_CATMULLROM;
    throw std::invalid_argument("Unsupported timm RepVGG interpolation: " + interpolation);
}

void validate_timm_repvgg_preprocess_config(const TimmRepvggPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("timm RepVGG input dimensions must be positive");
    }
    if (config.crop_pct <= 0.0F || config.crop_pct > 1.0F) {
        throw std::invalid_argument("timm RepVGG crop_pct must be in (0, 1]");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("timm RepVGG image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("timm RepVGG image std must be non-zero");
    }
}

int32_t torchvision_center_crop_offset(int32_t resized, int32_t target) {
    const int32_t difference = resized - target;
    const int32_t half = difference / 2;
    // torchvision uses Python round(), whose .5 ties round to the nearest even integer.
    return (difference % 2 != 0 && half % 2 != 0) ? half + 1 : half;
}

} // namespace

TimmRepvggResizeShape compute_timm_repvgg_resize_shape(int32_t image_height, int32_t image_width,
                                                       const TimmRepvggPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("timm RepVGG source dimensions must be positive");
    }
    validate_timm_repvgg_preprocess_config(config);

    if (config.input_image_h == config.input_image_w) {
        // timm's center-crop eval transform passes floor(input_size / crop_pct) as a scalar
        // torchvision Resize size. A scalar fixes the shorter edge and floors the aspect-ratio
        // calculation for the longer edge.
        const int32_t resized_short = static_cast<int32_t>(
            std::floor(static_cast<float>(config.input_image_h) / config.crop_pct));
        if (image_height <= image_width) {
            return {resized_short, static_cast<int32_t>(static_cast<int64_t>(resized_short) *
                                                        image_width / image_height)};
        }
        return {
            static_cast<int32_t>(static_cast<int64_t>(resized_short) * image_height / image_width),
            resized_short};
    }

    const float required_scale =
        std::max(static_cast<float>(config.input_image_h) / static_cast<float>(image_height),
                 static_cast<float>(config.input_image_w) / static_cast<float>(image_width));
    const float resize_scale = required_scale / config.crop_pct;
    return {
        std::max(config.input_image_h,
                 static_cast<int32_t>(std::floor(static_cast<float>(image_height) * resize_scale))),
        std::max(config.input_image_w,
                 static_cast<int32_t>(std::floor(static_cast<float>(image_width) * resize_scale))),
    };
}

std::vector<float> preprocess_timm_repvgg_image(const float* image_pixels, int32_t image_height,
                                                int32_t image_width,
                                                const TimmRepvggPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("timm RepVGG source image must be non-empty");
    }
    validate_timm_repvgg_preprocess_config(config);

    const auto resize_shape = compute_timm_repvgg_resize_shape(image_height, image_width, config);
    const int32_t resized_h = resize_shape.height;
    const int32_t resized_w = resize_shape.width;

    std::vector<float> resized(static_cast<std::size_t>(resized_h) * resized_w * 3U);
    if (stbir_resize(image_pixels, image_width, image_height,
                     image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(),
                     resized_w, resized_h, resized_w * 3 * static_cast<int32_t>(sizeof(float)),
                     STBIR_RGB, STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP,
                     resolve_timm_repvgg_resize_filter(config.interpolation)) == nullptr) {
        throw std::runtime_error("Failed to resize timm RepVGG input image");
    }

    const int32_t crop_y = torchvision_center_crop_offset(resized_h, config.input_image_h);
    const int32_t crop_x = torchvision_center_crop_offset(resized_w, config.input_image_w);
    const auto output_plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * output_plane);
    for (int32_t y = 0; y < config.input_image_h; ++y) {
        for (int32_t x = 0; x < config.input_image_w; ++x) {
            const auto src_idx =
                static_cast<std::size_t>((((crop_y + y) * resized_w + crop_x + x) * 3));
            for (int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                pixel_values[channel * output_plane +
                             static_cast<std::size_t>(y) * config.input_image_w + x] =
                    (resized[src_idx + channel] - config.image_mean[channel]) /
                    config.image_std[channel];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
