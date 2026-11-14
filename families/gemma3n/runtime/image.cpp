/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The antialiased bilinear resize reproduces the reference image processor,
// which resizes uint8 tensors with PyTorch's separable fixed-point kernel. The
// tap and int16 weight-precision computation is adapted from PyTorch v2.12.0,
// aten/src/ATen/native/cpu/UpSampleKernel.cpp
// (_compute_indices_min_size_weights_aa, _compute_index_ranges_int16_weights)
// and UpSampleKernelAVXAntialias.h, BSD-3-Clause. PyTorch derives that code
// from Pillow-SIMD's Resample.c (MIT-CMU License). The notices are reproduced
// in families/gemma3n/NOTICE.

#include "families/gemma3n/runtime/image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace trtmc::gemma3n {
namespace {

struct Taps {
    int64_t first{0};
    std::vector<int32_t> weights;
};

struct Plan {
    std::vector<Taps> rows;
    int32_t precision{0};
};

double triangle(double x) {
    x = std::abs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

// Normalized triangle-filter taps per output index, then int16 coefficients
// with the largest precision that keeps the largest weight representable.
Plan bilinear_plan(int64_t in_size, int64_t out_size) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double support = scale >= 1.0 ? scale : 1.0;
    const double inverse = scale >= 1.0 ? 1.0 / scale : 1.0;
    const auto max_taps = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
    std::vector<std::pair<int64_t, std::vector<double>>> taps(static_cast<std::size_t>(out_size));
    double largest = 0.0;
    for (int64_t i = 0; i < out_size; ++i) {
        const double center = scale * (static_cast<double>(i) + 0.5);
        const auto first = std::max(static_cast<int64_t>(center - support + 0.5), int64_t{0});
        auto count = std::min(static_cast<int64_t>(center + support + 0.5), in_size) - first;
        count = std::clamp(count, int64_t{0}, max_taps);
        std::vector<double> weights(static_cast<std::size_t>(count));
        double total = 0.0;
        for (int64_t j = 0; j < count; ++j) {
            weights[std::size_t(j)] =
                triangle((static_cast<double>(j + first) - center + 0.5) * inverse);
            total += weights[std::size_t(j)];
        }
        if (total != 0.0)
            for (auto& weight : weights) {
                weight /= total;
                largest = std::max(largest, weight);
            }
        taps[std::size_t(i)] = {first, std::move(weights)};
    }
    Plan plan;
    for (plan.precision = 0; plan.precision < 22; ++plan.precision) {
        const auto next =
            static_cast<int32_t>(0.5 + largest * static_cast<double>(1 << (plan.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    for (const auto& [first, weights] : taps) {
        Taps row{first, {}};
        for (const double weight : weights) {
            const double value = weight * static_cast<double>(1 << plan.precision);
            row.weights.push_back(static_cast<int32_t>(value < 0 ? -0.5 + value : 0.5 + value));
        }
        plan.rows.push_back(std::move(row));
    }
    return plan;
}

std::vector<uint8_t> resample_axis(const std::vector<uint8_t>& src, int32_t height, int32_t width,
                                   const Plan& plan, bool along_width) {
    const auto out_height = along_width ? height : static_cast<int32_t>(plan.rows.size());
    const auto out_width = along_width ? static_cast<int32_t>(plan.rows.size()) : width;
    std::vector<uint8_t> out(std::size_t(out_height) * std::size_t(out_width) * 3U);
    const int64_t half = plan.precision > 0 ? int64_t{1} << (plan.precision - 1) : 0;
    for (int32_t y = 0; y < out_height; ++y)
        for (int32_t x = 0; x < out_width; ++x) {
            const auto& row = plan.rows[std::size_t(along_width ? x : y)];
            for (int32_t c = 0; c < 3; ++c) {
                int64_t sum = half;
                for (std::size_t tap = 0; tap < row.weights.size(); ++tap) {
                    const auto source = static_cast<int64_t>(tap) + row.first;
                    const auto sy = along_width ? int64_t{y} : source;
                    const auto sx = along_width ? source : int64_t{x};
                    sum +=
                        int64_t{src[(std::size_t(sy) * std::size_t(width) + std::size_t(sx)) * 3U +
                                    std::size_t(c)]} *
                        row.weights[tap];
                }
                out[(std::size_t(y) * std::size_t(out_width) + std::size_t(x)) * 3U +
                    std::size_t(c)] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> plan.precision, 0, 255));
            }
        }
    return out;
}

} // namespace

Rgb8 to_rgb8(const internal::ImageView& image) {
    if (image.channels != 3)
        throw std::invalid_argument("Gemma 3n images must have three RGB channels");
    if (image.height == 0 || image.width == 0 || image.data == nullptr)
        throw std::invalid_argument("Gemma 3n images must be nonempty");
    const auto count = std::size_t(image.height) * std::size_t(image.width) * 3U;
    Rgb8 result{std::vector<uint8_t>(count), static_cast<int32_t>(image.height),
                static_cast<int32_t>(image.width)};
    if (image.format == internal::ImageFormat::UInt8) {
        if (image.byte_size != count)
            throw std::invalid_argument("Gemma 3n UInt8 image size does not match its shape");
        const auto* source = static_cast<const uint8_t*>(image.data);
        std::copy(source, source + count, result.pixels.begin());
    } else if (image.format == internal::ImageFormat::Float32) {
        if (image.byte_size != count * sizeof(float))
            throw std::invalid_argument("Gemma 3n Float32 image size does not match its shape");
        const auto* source = static_cast<const float*>(image.data);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(source[i]) || source[i] < 0.0F || source[i] > 1.0F)
                throw std::invalid_argument("Gemma 3n Float32 pixels must lie in [0, 1]");
            result.pixels[i] = static_cast<uint8_t>(std::lround(source[i] * 255.0F));
        }
    } else {
        throw std::invalid_argument("Unsupported Gemma 3n image format");
    }
    return result;
}

std::vector<uint8_t> resize_bilinear_aa(const std::vector<uint8_t>& rgb, int32_t height,
                                        int32_t width, int32_t out_height, int32_t out_width) {
    if (height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 ||
        rgb.size() != std::size_t(height) * std::size_t(width) * 3U)
        throw std::invalid_argument("Gemma 3n resize: invalid image geometry");
    auto image = rgb;
    if (out_width != width)
        image = resample_axis(image, height, width, bilinear_plan(width, out_width), true);
    if (out_height != height)
        image = resample_axis(image, height, out_width, bilinear_plan(height, out_height), false);
    return image;
}

std::vector<float> pixel_values(const Rgb8& image, int32_t size, float rescale) {
    const auto resized = resize_bilinear_aa(image.pixels, image.height, image.width, size, size);
    const auto plane = std::size_t(size) * std::size_t(size);
    std::vector<float> values(plane * 3U);
    for (std::size_t i = 0; i < plane; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            values[c * plane + i] = static_cast<float>(resized[i * 3U + c]) * rescale;
    return values;
}

} // namespace trtmc::gemma3n
