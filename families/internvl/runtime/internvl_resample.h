/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// 8-bit RGB resize that reproduces the reference image processor exactly: the
// GotOcr2 fast processor resizes uint8 tensors with PyTorch's antialiased
// bicubic interpolation, a separable fixed-point filter that rounds to 8 bits
// after the width pass and after the height pass.
//
// The tap and int16 weight-precision computation is adapted from PyTorch
// v2.12.0, aten/src/ATen/native/cpu/UpSampleKernel.cpp
// (_compute_indices_min_size_weights_aa and _compute_index_ranges_int16_weights),
// BSD-3-Clause. PyTorch derives that code from Pillow-SIMD's Resample.c
// (MIT-CMU License). The notices are reproduced in families/internvl/NOTICE.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {
namespace internvl {

struct ResampleTaps {
    int32_t first{0};
    std::vector<int32_t> weights;
};

struct ResamplePlan {
    std::vector<ResampleTaps> rows;
    int32_t precision{0};
};

// PyTorch's antialiased bicubic kernel (a = -0.5).
inline double torch_aa_bicubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

inline int32_t fixed_point(double weight, int32_t precision) {
    const double scaled = weight * static_cast<double>(1 << precision);
    return static_cast<int32_t>(weight < 0.0 ? -0.5 + scaled : 0.5 + scaled);
}

// Normalized filter taps for every output position, as PyTorch's
// _compute_indices_weights_aa computes them.
template <typename Filter>
inline std::vector<std::pair<int32_t, std::vector<double>>>
resample_taps(int32_t in_size, int32_t out_size, double support_radius, Filter filter) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support = support_radius * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<std::pair<int32_t, std::vector<double>>> taps(static_cast<std::size_t>(out_size));
    for (int32_t index = 0; index < out_size; ++index) {
        const double center = (static_cast<double>(index) + 0.5) * scale;
        const auto first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const auto count = std::min(static_cast<int32_t>(center + support + 0.5), in_size) - first;
        std::vector<double> weights(static_cast<std::size_t>(std::max(count, 0)));
        double total = 0.0;
        for (int32_t tap = 0; tap < count; ++tap) {
            weights[static_cast<std::size_t>(tap)] =
                filter((static_cast<double>(tap + first) - center + 0.5) * inverse_filter_scale);
            total += weights[static_cast<std::size_t>(tap)];
        }
        if (total != 0.0)
            for (auto& weight : weights)
                weight /= total;
        taps[static_cast<std::size_t>(index)] = {first, std::move(weights)};
    }
    return taps;
}

// PyTorch: int16 coefficients with the largest precision that keeps the
// biggest weight representable.
inline ResamplePlan torch_aa_bicubic_plan(int32_t in_size, int32_t out_size) {
    const auto taps = resample_taps(in_size, out_size, 2.0, torch_aa_bicubic);
    double largest = 0.0;
    for (const auto& [first, weights] : taps)
        for (const double weight : weights)
            largest = std::max(largest, std::abs(weight));
    ResamplePlan plan;
    while (plan.precision < 22 &&
           static_cast<int32_t>(0.5 + largest * static_cast<double>(1 << (plan.precision + 1))) <
               (1 << 15))
        ++plan.precision;
    for (const auto& [first, weights] : taps) {
        ResampleTaps row{first, {}};
        for (const double weight : weights)
            row.weights.push_back(fixed_point(weight, plan.precision));
        plan.rows.push_back(std::move(row));
    }
    return plan;
}

// Resample one axis of an HWC uint8 RGB image, rounding to 8 bits.
inline std::vector<uint8_t> resample_rgb_axis(const std::vector<uint8_t>& src, int32_t height,
                                              int32_t width, const ResamplePlan& plan,
                                              bool along_width) {
    const auto out_height = along_width ? height : static_cast<int32_t>(plan.rows.size());
    const auto out_width = along_width ? static_cast<int32_t>(plan.rows.size()) : width;
    std::vector<uint8_t> out(static_cast<std::size_t>(out_height) * out_width * 3U);
    const int64_t half = int64_t{1} << (plan.precision - 1);
    for (int32_t y = 0; y < out_height; ++y) {
        for (int32_t x = 0; x < out_width; ++x) {
            const auto& row = plan.rows[static_cast<std::size_t>(along_width ? x : y)];
            for (int32_t c = 0; c < 3; ++c) {
                int64_t sum = half;
                for (std::size_t tap = 0; tap < row.weights.size(); ++tap) {
                    const auto source = static_cast<int32_t>(tap) + row.first;
                    const auto sy = along_width ? y : source;
                    const auto sx = along_width ? source : x;
                    sum += static_cast<int64_t>(
                               src[(static_cast<std::size_t>(sy) * width + sx) * 3U + c]) *
                           row.weights[tap];
                }
                out[(static_cast<std::size_t>(y) * out_width + x) * 3U + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> plan.precision, 0, 255));
            }
        }
    }
    return out;
}

// F.interpolate(uint8 NCHW, mode="bicubic", align_corners=False,
// antialias=True) on CPU: the width pass first, then the height pass.
inline std::vector<uint8_t> torch_aa_bicubic_resize_rgb(const std::vector<uint8_t>& src,
                                                        int32_t height, int32_t width,
                                                        int32_t out_height, int32_t out_width) {
    if (height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 ||
        src.size() != static_cast<std::size_t>(height) * width * 3U)
        throw std::invalid_argument("torch_aa_bicubic_resize_rgb: invalid image geometry");
    auto image = src;
    if (out_width != width)
        image =
            resample_rgb_axis(image, height, width, torch_aa_bicubic_plan(width, out_width), true);
    if (out_height != height)
        image = resample_rgb_axis(image, height, out_width,
                                  torch_aa_bicubic_plan(height, out_height), false);
    return image;
}

} // namespace internvl
} // namespace trtmc
