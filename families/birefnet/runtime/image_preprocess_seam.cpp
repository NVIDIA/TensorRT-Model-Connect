/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/birefnet/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc {
namespace {

float sample_bilinear(const float* pixels, int32_t height, int32_t width, int32_t channel, float y,
                      float x) {
    const float clamped_y = std::min(std::max(y, 0.0F), static_cast<float>(height - 1));
    const float clamped_x = std::min(std::max(x, 0.0F), static_cast<float>(width - 1));
    const auto y0 = static_cast<int32_t>(std::floor(clamped_y));
    const auto x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y1 = std::min(y0 + 1, height - 1);
    const int32_t x1 = std::min(x0 + 1, width - 1);
    const float dy = clamped_y - static_cast<float>(y0);
    const float dx = clamped_x - static_cast<float>(x0);
    const auto at = [&](int32_t row, int32_t column) {
        return pixels[(static_cast<std::size_t>(row) * width + column) * 3 + channel];
    };
    const float top = at(y0, x0) * (1.0F - dx) + at(y0, x1) * dx;
    const float bottom = at(y1, x0) * (1.0F - dx) + at(y1, x1) * dx;
    return top * (1.0F - dy) + bottom * dy;
}

} // namespace

std::vector<float> preprocess_birefnet_image(const float* pixels, int32_t image_height,
                                             int32_t image_width,
                                             const BiRefNetPreprocessConfig& config) {
    if (pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("birefnet preprocessing received an empty image");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("birefnet preprocessing needs three-channel statistics");

    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;
    std::vector<float> out(static_cast<std::size_t>(3) * out_h * out_w);
    const float scale_y = static_cast<float>(image_height) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(image_width) / static_cast<float>(out_w);

    for (int32_t channel = 0; channel < 3; ++channel) {
        const float mean = config.image_mean[static_cast<std::size_t>(channel)];
        const float deviation = config.image_std[static_cast<std::size_t>(channel)];
        if (deviation == 0.0F)
            throw std::invalid_argument("birefnet preprocessing standard deviation is zero");
        for (int32_t row = 0; row < out_h; ++row) {
            const float y = (static_cast<float>(row) + 0.5F) * scale_y - 0.5F;
            for (int32_t column = 0; column < out_w; ++column) {
                const float x = (static_cast<float>(column) + 0.5F) * scale_x - 0.5F;
                // The reference resizes in 8-bit before converting to float.
                const float raw = sample_bilinear(pixels, image_height, image_width, channel, y, x);
                const float quantised =
                    std::min(255.0F, std::max(0.0F, std::round(raw * 255.0F))) / 255.0F;
                out[(static_cast<std::size_t>(channel) * out_h + row) * out_w + column] =
                    (quantised - mean) / deviation;
            }
        }
    }
    return out;
}

std::vector<int32_t> mask_from_logits(const float* logits, int32_t logit_height,
                                      int32_t logit_width, int32_t out_height, int32_t out_width,
                                      float threshold) {
    if (logits == nullptr || logit_height <= 0 || logit_width <= 0)
        throw std::invalid_argument("birefnet received an empty logit map");
    if (out_height <= 0 || out_width <= 0)
        throw std::invalid_argument("birefnet mask target size must be positive");

    std::vector<int32_t> mask(static_cast<std::size_t>(out_height) * out_width);
    const float scale_y = static_cast<float>(logit_height) / static_cast<float>(out_height);
    const float scale_x = static_cast<float>(logit_width) / static_cast<float>(out_width);
    for (int32_t row = 0; row < out_height; ++row) {
        const auto source_row = std::min(
            logit_height - 1, static_cast<int32_t>((static_cast<float>(row) + 0.5F) * scale_y));
        for (int32_t column = 0; column < out_width; ++column) {
            const auto source_column =
                std::min(logit_width - 1,
                         static_cast<int32_t>((static_cast<float>(column) + 0.5F) * scale_x));
            const float logit =
                logits[static_cast<std::size_t>(source_row) * logit_width + source_column];
            const float probability = 1.0F / (1.0F + std::exp(-logit));
            mask[static_cast<std::size_t>(row) * out_width + column] =
                probability > threshold ? 1 : 0;
        }
    }
    return mask;
}

} // namespace trtmc
