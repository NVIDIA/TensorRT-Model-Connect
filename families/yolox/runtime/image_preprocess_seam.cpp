/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolox/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc {
namespace {

float sample_bilinear(const float* image, std::int32_t height, std::int32_t width,
                      std::int32_t channel, float y, float x) {
    // The task supplies interleaved RGB floats. YOLOX resizes integer bytes,
    // so quantize the four input samples before interpolation.
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    const auto y0 = static_cast<std::int32_t>(clamped_y);
    const auto x0 = static_cast<std::int32_t>(clamped_x);
    const std::int32_t y1 = std::min(y0 + 1, height - 1);
    const std::int32_t x1 = std::min(x0 + 1, width - 1);
    const double wy = static_cast<double>(clamped_y) - y0;
    const double wx = static_cast<double>(clamped_x) - x0;
    const auto at = [&](std::int32_t row, std::int32_t column) {
        const auto index = (static_cast<std::size_t>(row) * width + column) * 3U + channel;
        const float value = image[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("YOLOX input must be finite RGB pixels in [0, 1]");
        return std::round(value * 255.0F);
    };
    const double top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
    const double bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
    // OpenCV's fixed-point byte resize may differ by one; E2E bounds this.
    return static_cast<float>(std::round(top * (1.0 - wy) + bottom * wy));
}

} // namespace

std::vector<float> preprocess_yolox_image(const float* pixels, std::int32_t height,
                                          std::int32_t width, const YoloxPreprocessConfig& config,
                                          YoloxLetterbox& letterbox) {
    if (pixels == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("YOLOX preprocessing needs a non-empty image");
    if (config.input_image_h <= 0 || config.input_image_w <= 0 ||
        !std::isfinite(config.pad_value) || config.pad_value < 0.0F || config.pad_value > 255.0F)
        throw std::invalid_argument("YOLOX preprocessing configuration is invalid");

    // Upstream non-legacy preproc: truncate resized dimensions, paste at the
    // top left, pad the right and bottom with 114, keep BGR bytes without /255.
    const double scale = std::min(static_cast<double>(config.input_image_h) / height,
                                  static_cast<double>(config.input_image_w) / width);
    const auto scaled_h = static_cast<std::int32_t>(height * scale);
    const auto scaled_w = static_cast<std::int32_t>(width * scale);
    if (scaled_h < 1 || scaled_w < 1)
        throw std::invalid_argument("YOLOX image aspect ratio produces an empty resize");
    letterbox = {static_cast<float>(scale), 0.0F, 0.0F};
    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> values(3U * plane, config.pad_value);
    for (std::int32_t channel = 0; channel < 3; ++channel) {
        float* target = values.data() + static_cast<std::size_t>(channel) * plane;
        for (std::int32_t row = 0; row < scaled_h; ++row) {
            const float y = static_cast<float>((row + 0.5) * height / scaled_h - 0.5);
            for (std::int32_t column = 0; column < scaled_w; ++column) {
                const float x = static_cast<float>((column + 0.5) * width / scaled_w - 0.5);
                // Select the RGB source channel for this BGR output plane.
                target[static_cast<std::size_t>(row) * config.input_image_w + column] =
                    sample_bilinear(pixels, height, width, 2 - channel, y, x);
            }
        }
    }
    return values;
}

} // namespace trtmc
