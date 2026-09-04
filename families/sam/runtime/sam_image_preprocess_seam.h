/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/sam/runtime/decoded_image.h"
#include "families/sam/runtime/sam_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace trtmc {

struct SamImageEncodePlan {
    std::vector<float> pixel_values;
    int32_t rescaled_width{0};
    int32_t rescaled_height{0};
    int32_t original_width{0};
    int32_t original_height{0};

    [[nodiscard]] bool ok() const {
        return !pixel_values.empty() && rescaled_width > 0 && rescaled_height > 0 &&
               original_width > 0 && original_height > 0;
    }
};

inline float sample_image_channel_bilinear(const runtime::adapters::io::DecodedImage& image,
                                           int32_t channel, float x, float y) {
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(image.width - 1));
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(image.height - 1));
    const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
    const int32_t x1 = std::min(x0 + 1, image.width - 1);
    const int32_t y1 = std::min(y0 + 1, image.height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const auto idx = [stride = image.channels, width = image.width](int32_t yy, int32_t xx,
                                                                    int32_t cc) {
        return static_cast<std::size_t>((yy * width + xx) * stride + cc);
    };

    const float v00 = static_cast<float>(image.pixels[idx(y0, x0, channel)]);
    const float v01 = static_cast<float>(image.pixels[idx(y0, x1, channel)]);
    const float v10 = static_cast<float>(image.pixels[idx(y1, x0, channel)]);
    const float v11 = static_cast<float>(image.pixels[idx(y1, x1, channel)]);
    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bottom = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

inline SamImageEncodePlan
build_sam_image_encode_plan(const runtime::adapters::io::DecodedImage& image,
                            const SamConfig& config) {
    SamImageEncodePlan plan;
    if (image.empty()) {
        return plan;
    }

    const int32_t target_h = config.image_size;
    const int32_t target_w = config.image_size;
    const int32_t longest_side = std::max(image.width, image.height);
    const float scale = static_cast<float>(target_h) / static_cast<float>(longest_side);

    plan.rescaled_width = static_cast<int32_t>(std::round(static_cast<float>(image.width) * scale));
    plan.rescaled_height =
        static_cast<int32_t>(std::round(static_cast<float>(image.height) * scale));
    plan.original_width = image.width;
    plan.original_height = image.height;
    plan.pixel_values.assign(static_cast<std::size_t>(3) * target_h * target_w, 0.0F);

    for (int32_t y = 0; y < plan.rescaled_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(image.height) /
                                static_cast<float>(plan.rescaled_height) -
                            0.5F;
        for (int32_t x = 0; x < plan.rescaled_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(image.width) /
                                    static_cast<float>(plan.rescaled_width) -
                                0.5F;
            for (int32_t c = 0; c < 3; ++c) {
                float value = sample_image_channel_bilinear(image, c, src_x, src_y) / 255.0F;
                value = (value - config.image_mean[c]) / config.image_std[c];
                plan.pixel_values[static_cast<std::size_t>(c) * target_h * target_w +
                                  static_cast<std::size_t>(y) * target_w + x] = value;
            }
        }
    }

    return plan;
}

} // namespace trtmc
