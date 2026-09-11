/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/detr/runtime/image_preprocess_seam.h"

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

void validate_detr_preprocess_config(const DetrPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("detr input dimensions must be positive");
    }
    if (config.shortest_edge <= 0 || config.longest_edge <= 0) {
        throw std::invalid_argument("detr resize edges must be positive");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("detr image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F) {
            throw std::invalid_argument("detr image std must be non-zero");
        }
    }
}

int32_t hf_round(double value) {
    return static_cast<int32_t>(std::round(value));
}

struct DetrResizePlan {
    double raw_size{0.0};
    bool has_raw_size{false};
    int32_t short_edge{0};
};

DetrResizePlan compute_detr_resize_plan(double height, double width,
                                        const DetrPreprocessConfig& config) {
    const double size = static_cast<double>(config.shortest_edge);
    const double max_size = static_cast<double>(config.longest_edge);
    const double min_original = std::min(height, width);
    const double max_original = std::max(height, width);
    DetrResizePlan plan;
    plan.short_edge = config.shortest_edge;
    if (max_original / min_original * size > max_size) {
        plan.raw_size = max_size * min_original / max_original;
        plan.has_raw_size = true;
        plan.short_edge = hf_round(plan.raw_size);
    }
    return plan;
}

int32_t resize_other_axis(const DetrResizePlan& plan, double size, double primary,
                          double secondary) {
    if (plan.has_raw_size) {
        return static_cast<int32_t>(plan.raw_size * secondary / primary);
    }
    return static_cast<int32_t>(size * secondary / primary);
}

} // namespace

DetrResizeShape compute_detr_resize_shape(int32_t image_height, int32_t image_width,
                                          const DetrPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("detr source dimensions must be positive");
    }
    validate_detr_preprocess_config(config);

    const double size = static_cast<double>(config.shortest_edge);
    const double height = static_cast<double>(image_height);
    const double width = static_cast<double>(image_width);
    const DetrResizePlan plan = compute_detr_resize_plan(height, width, config);

    if (std::min(height, width) == plan.short_edge) {
        return {image_height, image_width};
    }
    if (width < height) {
        return {resize_other_axis(plan, size, width, height), plan.short_edge};
    }
    return {plan.short_edge, resize_other_axis(plan, size, height, width)};
}

std::vector<float> preprocess_detr_image(const float* image_pixels, int32_t image_height,
                                         int32_t image_width, const DetrPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("detr source image must be non-empty");
    }
    validate_detr_preprocess_config(config);

    const auto resize_shape = compute_detr_resize_shape(image_height, image_width, config);
    const int32_t resized_h = resize_shape.height;
    const int32_t resized_w = resize_shape.width;

    std::vector<float> resized(static_cast<std::size_t>(resized_h) * resized_w * 3U);
    if (stbir_resize(image_pixels, image_width, image_height,
                     image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(),
                     resized_w, resized_h, resized_w * 3 * static_cast<int32_t>(sizeof(float)),
                     STBIR_RGB, STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_TRIANGLE) == nullptr) {
        throw std::runtime_error("Failed to resize detr input image");
    }

    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;
    if (resized_h > out_h || resized_w > out_w) {
        throw std::invalid_argument(
            "detr input resize dimensions exceed the engine input dimensions");
    }
    const auto output_plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> pixel_values(3U * output_plane, 0.0F);

    for (int32_t y = 0; y < out_h; ++y) {
        for (int32_t x = 0; x < out_w; ++x) {
            if (y >= resized_h || x >= resized_w)
                continue;
            const auto src_idx = static_cast<std::size_t>(((y * resized_w) + x) * 3);
            const float r = (resized[src_idx] - config.image_mean[0]) / config.image_std[0];
            const float g = (resized[src_idx + 1] - config.image_mean[1]) / config.image_std[1];
            const float b = (resized[src_idx + 2] - config.image_mean[2]) / config.image_std[2];
            pixel_values[static_cast<std::size_t>(y) * out_w + x] = r;
            pixel_values[output_plane + static_cast<std::size_t>(y) * out_w + x] = g;
            pixel_values[2U * output_plane + static_cast<std::size_t>(y) * out_w + x] = b;
        }
    }
    return pixel_values;
}

} // namespace trtmc
