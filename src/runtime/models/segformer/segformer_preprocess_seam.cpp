/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/segformer/segformer_preprocess_seam.h"

#include "stb_image_resize2.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

void validate_segformer_preprocess_config(const SegformerPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("SegFormer input dimensions must be positive");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("SegFormer image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("SegFormer image std must be non-zero");
    }
}

} // namespace

std::vector<float> preprocess_segformer_image(const float* image_pixels, int32_t image_height,
                                              int32_t image_width,
                                              const SegformerPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("SegFormer source image must be non-empty");
    }
    validate_segformer_preprocess_config(config);

    const int32_t input_h = config.input_image_h;
    const int32_t input_w = config.input_image_w;
    std::vector<float> resized(static_cast<std::size_t>(input_h) * input_w * 3U);
    if (stbir_resize(image_pixels, image_width, image_height,
                     image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(), input_w,
                     input_h, input_w * 3 * static_cast<int32_t>(sizeof(float)), STBIR_RGB,
                     STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE) == nullptr) {
        throw std::runtime_error("Failed to resize SegFormer input image");
    }

    std::vector<float> pixel_values(static_cast<std::size_t>(3) * input_h * input_w);
    for (int32_t y = 0; y < input_h; ++y) {
        for (int32_t x = 0; x < input_w; ++x) {
            const auto src_idx = static_cast<std::size_t>((y * input_w + x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                const float value = (resized[src_idx + channel] - config.image_mean[channel]) /
                                    config.image_std[channel];
                pixel_values[channel * static_cast<std::size_t>(input_h) * input_w +
                             static_cast<std::size_t>(y) * input_w + x] = value;
            }
        }
    }
    return pixel_values;
}

std::vector<float> preprocess_segformer_image(const runtime::adapters::io::DecodedImage& image,
                                              const SegformerPreprocessConfig& config) {
    if (image.empty() || image.channels < 3) {
        throw std::invalid_argument("SegFormer decoded image must contain RGB pixels");
    }

    std::vector<float> pixels(static_cast<std::size_t>(image.height) * image.width * 3U);
    for (int32_t y = 0; y < image.height; ++y) {
        for (int32_t x = 0; x < image.width; ++x) {
            const auto src_idx = static_cast<std::size_t>((y * image.width + x) * image.channels);
            const auto dst_idx = static_cast<std::size_t>((y * image.width + x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                pixels[dst_idx + static_cast<std::size_t>(c)] =
                    static_cast<float>(image.pixels[src_idx + static_cast<std::size_t>(c)]) /
                    255.0F;
            }
        }
    }
    return preprocess_segformer_image(pixels.data(), image.height, image.width, config);
}

} // namespace trtmc
