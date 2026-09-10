/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolov10/runtime/image_preprocess_seam.h"

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

std::vector<float> preprocess_yolov10_image(const float* pixels, std::int32_t height,
                                            std::int32_t width,
                                            const Yolov10PreprocessConfig& config,
                                            Yolov10Letterbox& letterbox) {
    if (pixels == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("YOLOv10 preprocessing needs a non-empty image");
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("YOLOv10 input size must be positive");

    // Fit the longest side, centre the result, and pad the rest. Cropping
    // instead would drop objects at the border.
    const float scale =
        std::min(static_cast<float>(config.input_image_h) / static_cast<float>(height),
                 static_cast<float>(config.input_image_w) / static_cast<float>(width));
    const auto scaled_h =
        static_cast<std::int32_t>(std::lround(static_cast<float>(height) * scale));
    const auto scaled_w = static_cast<std::int32_t>(std::lround(static_cast<float>(width) * scale));
    const float pad_y = static_cast<float>(config.input_image_h - scaled_h) / 2.0F;
    const float pad_x = static_cast<float>(config.input_image_w - scaled_w) / 2.0F;
    letterbox.scale = scale;
    letterbox.pad_x = pad_x;
    letterbox.pad_y = pad_y;

    const auto plane = static_cast<std::size_t>(config.input_image_h) *
                       static_cast<std::size_t>(config.input_image_w);
    std::vector<float> values(plane * 3U, config.pad_value);
    const auto top = static_cast<std::int32_t>(pad_y);
    const auto left = static_cast<std::int32_t>(pad_x);
    for (std::int32_t channel = 0; channel < 3; ++channel) {
        float* target = values.data() + static_cast<std::size_t>(channel) * plane;
        for (std::int32_t row = 0; row < scaled_h; ++row) {
            const float y = (static_cast<float>(row) + 0.5F) / scale - 0.5F;
            for (std::int32_t column = 0; column < scaled_w; ++column) {
                const float x = (static_cast<float>(column) + 0.5F) / scale - 0.5F;
                target[static_cast<std::size_t>(top + row) *
                           static_cast<std::size_t>(config.input_image_w) +
                       static_cast<std::size_t>(left + column)] =
                    sample_bilinear(pixels, height, width, channel, y, x);
            }
        }
    }
    return values;
}

} // namespace trtmc
