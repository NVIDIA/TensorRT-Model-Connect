/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deformable_detr/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc {
namespace {

// Half-pixel centred bilinear sampling, matching the reference resize.
float sample_bilinear(const float* pixels, std::int32_t height, std::int32_t width,
                      std::int32_t channels, std::int32_t channel, float y, float x) {
    const float clamped_y = std::min(std::max(y, 0.0F), static_cast<float>(height - 1));
    const float clamped_x = std::min(std::max(x, 0.0F), static_cast<float>(width - 1));
    const auto y0 = static_cast<std::int32_t>(std::floor(clamped_y));
    const auto x0 = static_cast<std::int32_t>(std::floor(clamped_x));
    const std::int32_t y1 = std::min(y0 + 1, height - 1);
    const std::int32_t x1 = std::min(x0 + 1, width - 1);
    const float dy = clamped_y - static_cast<float>(y0);
    const float dx = clamped_x - static_cast<float>(x0);

    const auto at = [&](std::int32_t row, std::int32_t column) {
        return pixels[(static_cast<std::size_t>(row) * width + column) * channels + channel];
    };
    const float top = at(y0, x0) * (1.0F - dx) + at(y0, x1) * dx;
    const float bottom = at(y1, x0) * (1.0F - dx) + at(y1, x1) * dx;
    return top * (1.0F - dy) + bottom * dy;
}

} // namespace

std::vector<float> preprocess_deformable_detr_image(const float* pixels, std::int32_t image_height,
                                                    std::int32_t image_width,
                                                    const DeformableDetrPreprocessConfig& config) {
    if (pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("deformable_detr preprocessing received an empty image");
    const std::int32_t out_h = config.input_image_h;
    const std::int32_t out_w = config.input_image_w;
    if (out_h <= 0 || out_w <= 0)
        throw std::invalid_argument("deformable_detr preprocessing target size must be positive");

    constexpr std::int32_t kChannels = 3;
    std::vector<float> out(static_cast<std::size_t>(kChannels) * out_h * out_w);
    const float scale_y = static_cast<float>(image_height) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(image_width) / static_cast<float>(out_w);

    for (std::int32_t channel = 0; channel < kChannels; ++channel) {
        const float mean = config.image_mean[static_cast<std::size_t>(channel)];
        const float deviation = config.image_std[static_cast<std::size_t>(channel)];
        if (config.do_normalize && deviation == 0.0F)
            throw std::invalid_argument("deformable_detr image_std must be non-zero");
        for (std::int32_t row = 0; row < out_h; ++row) {
            const float y = (static_cast<float>(row) + 0.5F) * scale_y - 0.5F;
            for (std::int32_t column = 0; column < out_w; ++column) {
                const float x = (static_cast<float>(column) + 0.5F) * scale_x - 0.5F;
                const float value =
                    sample_bilinear(pixels, image_height, image_width, kChannels, channel, y, x);
                // The reference resizes in 8-bit and only then converts to
                // float, so the rounding is part of the contract rather than
                // an optimisation.
                const float quantised =
                    std::min(255.0F, std::max(0.0F, std::round(value * 255.0F))) / 255.0F;
                out[(static_cast<std::size_t>(channel) * out_h + row) * out_w + column] =
                    config.do_normalize ? (quantised - mean) / deviation : quantised;
            }
        }
    }
    return out;
}

} // namespace trtmc
