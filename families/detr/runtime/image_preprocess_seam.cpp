/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/detr/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc {
namespace {

float sample_bilinear(const float* image, std::int32_t height, std::int32_t width,
                      std::int32_t channel, float y, float x) {
    // The caller hands over an interleaved RGB image, so one step in x moves
    // three floats and the channel is an offset inside the pixel.
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    const auto y0 = static_cast<std::int32_t>(clamped_y);
    const auto x0 = static_cast<std::int32_t>(clamped_x);
    const std::int32_t y1 = std::min(y0 + 1, height - 1);
    const std::int32_t x1 = std::min(x0 + 1, width - 1);
    const float wy = clamped_y - static_cast<float>(y0);
    const float wx = clamped_x - static_cast<float>(x0);
    const auto at = [&](std::int32_t row, std::int32_t column) {
        return image[(static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(column)) *
                         3U +
                     static_cast<std::size_t>(channel)];
    };
    const float top = at(y0, x0) * (1.0F - wx) + at(y0, x1) * wx;
    const float bottom = at(y1, x0) * (1.0F - wx) + at(y1, x1) * wx;
    return top * (1.0F - wy) + bottom * wy;
}

} // namespace

std::vector<float> preprocess_detr_image(const float* pixels, std::int32_t height,
                                         std::int32_t width, const DetrPreprocessConfig& config) {
    if (pixels == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("DETR preprocessing needs a non-empty image");
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("DETR input size must be positive");
    if (config.image_mean.size() != 3U || config.image_std.size() != 3U)
        throw std::invalid_argument("DETR normalisation needs three channels");
    for (const float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("DETR normalisation std must be non-zero");
    }

    const auto out_h = config.input_image_h;
    const auto out_w = config.input_image_w;
    const auto plane = static_cast<std::size_t>(out_h) * static_cast<std::size_t>(out_w);
    std::vector<float> values(plane * 3U);

    const float scale_y = static_cast<float>(height) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(width) / static_cast<float>(out_w);
    for (std::int32_t channel = 0; channel < 3; ++channel) {
        const float mean = config.image_mean[static_cast<std::size_t>(channel)];
        const float inverse_std = 1.0F / config.image_std[static_cast<std::size_t>(channel)];
        float* target = values.data() + static_cast<std::size_t>(channel) * plane;
        for (std::int32_t y = 0; y < out_h; ++y) {
            // Sample at pixel centres so the two edges stay symmetric.
            const float source_y = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
            for (std::int32_t x = 0; x < out_w; ++x) {
                const float source_x = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
                const float sample =
                    sample_bilinear(pixels, height, width, channel, source_y, source_x);
                target[static_cast<std::size_t>(y) * static_cast<std::size_t>(out_w) +
                       static_cast<std::size_t>(x)] = (sample - mean) * inverse_std;
            }
        }
    }
    return values;
}

} // namespace trtmc
