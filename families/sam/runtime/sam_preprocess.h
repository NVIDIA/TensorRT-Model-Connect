/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SAM image preprocess: resize the longest side to image_size (bilinear,
// half-pixel centers), normalize each channel, and zero-pad to a square
// image_size x image_size CHW tensor.

#include "families/sam/runtime/sam_bilinear.h"
#include "families/sam/runtime/sam_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct SamPreprocessPlan {
    std::vector<float> pixel_values;
    int32_t rescaled_width{0};
    int32_t rescaled_height{0};
    int32_t original_width{0};
    int32_t original_height{0};
};

inline std::array<float, 3> sam_channel_values(const std::vector<float>& values,
                                               std::array<float, 3> defaults) {
    const auto count = std::min(values.size(), defaults.size());
    for (std::size_t i = 0; i < count; ++i)
        defaults[i] = values[i];
    return defaults;
}

// Destination index i of a `dst_len`-sample axis maps to this source
// coordinate on a `src_len`-sample axis.
inline float sam_preprocess_source_coordinate(int32_t i, int32_t src_len, int32_t dst_len) {
    return (static_cast<float>(i) + 0.5F) * static_cast<float>(src_len) /
               static_cast<float>(dst_len) -
           0.5F;
}

// Writes the rescaled region of plan.pixel_values from an HWC RGB float image:
// for each pixel and channel, the bilinear sample (sam_bilinear.h) minus the
// channel mean, divided by the channel std. Each source row is interpolated
// horizontally once for all three channels; the vertical blend and the
// normalization then run over contiguous rows.
inline void fill_sam_preprocessed_pixels(SamPreprocessPlan& plan, const float* image_pixels,
                                         int32_t height, int32_t width, const SamConfig& config,
                                         const std::array<float, 3>& mean,
                                         const std::array<float, 3>& stdv) {
    const int32_t rescaled_w = plan.rescaled_width;
    const int32_t rescaled_h = plan.rescaled_height;
    const auto x_axis = make_sam_bilinear_axis(width, rescaled_w, [&](int32_t x) {
        return sam_preprocess_source_coordinate(x, width, rescaled_w);
    });
    const auto y_axis = make_sam_bilinear_axis(height, rescaled_h, [&](int32_t y) {
        return sam_preprocess_source_coordinate(y, height, rescaled_h);
    });

    const auto row_w = static_cast<std::size_t>(rescaled_w);
    const auto plane =
        static_cast<std::size_t>(config.image_size) * static_cast<std::size_t>(config.image_size);
    // A cached source row holds its three channels' horizontal passes, planar.
    SamBilinearRowCache cache(3 * row_w);
    const auto fill = [&](int32_t row, float* out) {
        const float* src =
            image_pixels + static_cast<std::size_t>(row) * static_cast<std::size_t>(width) * 3;
        const int32_t* lo = x_axis.lo.data();
        const int32_t* hi = x_axis.hi.data();
        const float* weight = x_axis.weight.data();
        const float* inverse_weight = x_axis.inverse_weight.data();
        for (std::size_t x = 0; x < row_w; ++x) {
            const float* left = src + static_cast<std::size_t>(lo[x]) * 3;
            const float* right = src + static_cast<std::size_t>(hi[x]) * 3;
            for (std::size_t c = 0; c < 3; ++c) {
                out[c * row_w + x] = left[c] * inverse_weight[x] + right[c] * weight[x];
            }
        }
    };

    for (int32_t y = 0; y < rescaled_h; ++y) {
        const float* top = cache.row(y_axis.lo[y], y_axis.hi[y], fill);
        const float* bottom = cache.row(y_axis.hi[y], y_axis.lo[y], fill);
        const float weight = y_axis.weight[y];
        const float inverse_weight = y_axis.inverse_weight[y];
        for (std::size_t c = 0; c < 3; ++c) {
            const float* top_c = top + c * row_w;
            const float* bottom_c = bottom + c * row_w;
            const float channel_mean = mean[c];
            const float channel_std = stdv[c];
            float* out = plan.pixel_values.data() + c * plane +
                         static_cast<std::size_t>(y) * static_cast<std::size_t>(config.image_size);
            for (std::size_t x = 0; x < row_w; ++x) {
                out[x] =
                    (top_c[x] * inverse_weight + bottom_c[x] * weight - channel_mean) / channel_std;
            }
        }
    }
}

inline SamPreprocessPlan build_sam_preprocess_plan(const float* image_pixels, int32_t height,
                                                   int32_t width, const SamConfig& config) {
    SamPreprocessPlan plan;
    if (image_pixels == nullptr || height <= 0 || width <= 0 || config.image_size <= 0)
        return plan;

    const int32_t image_size = config.image_size;
    const int32_t longest_side = std::max(width, height);
    const float scale = static_cast<float>(image_size) / static_cast<float>(longest_side);
    plan.rescaled_width =
        std::max(1, static_cast<int32_t>(std::round(static_cast<float>(width) * scale)));
    plan.rescaled_height =
        std::max(1, static_cast<int32_t>(std::round(static_cast<float>(height) * scale)));
    plan.original_width = width;
    plan.original_height = height;
    plan.pixel_values.assign(static_cast<std::size_t>(3) * static_cast<std::size_t>(image_size) *
                                 static_cast<std::size_t>(image_size),
                             0.0F);

    const auto mean = sam_channel_values(config.image_mean, {0.485F, 0.456F, 0.406F});
    const auto stdv = sam_channel_values(config.image_std, {0.229F, 0.224F, 0.225F});
    fill_sam_preprocessed_pixels(plan, image_pixels, height, width, config, mean, stdv);

    return plan;
}

} // namespace trtmc
