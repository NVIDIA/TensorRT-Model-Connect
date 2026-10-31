/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The resampling numerical contract follows Pillow 12.2.0's Resample.c:
// https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/Resample.c
// Pillow's copyright and MIT-CMU permission notice are preserved in ../NOTICE.
// This family implementation uses a bounded rolling intermediate row window.

#include "families/minimax_h3/runtime/reference_image.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kBits = 22;
constexpr int64_t kUnit = int64_t{1} << kBits;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::size_t image_bytes(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("MiniMax-H3 reference image must have positive dimensions");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    if (w > std::numeric_limits<std::size_t>::max() / 3 / h)
        throw std::overflow_error("MiniMax-H3 reference RGB dimensions overflow");
    return h * w * 3;
}

int32_t aligned_even_round(double value) {
    const auto lower = static_cast<int32_t>(std::floor(value));
    const double fraction = value - lower;
    return std::max(1, lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2))) * 32;
}

double sinc(double value) {
    if (value == 0.0)
        return 1.0;
    value *= kPi;
    return std::sin(value) / value;
}

double lanczos(double value) {
    return value >= -3.0 && value < 3.0 ? sinc(value) * sinc(value / 3.0) : 0.0;
}

struct Taps {
    int32_t start;
    std::vector<int32_t> weights;
};

std::vector<Taps> coefficients(int32_t input, int32_t output) {
    // Pillow stores the full input box's right/bottom coordinate as float.
    const double scale = static_cast<double>(static_cast<float>(input)) / output;
    const double filter_scale = std::max(1.0, scale);
    const double support = 3.0 * filter_scale;
    const double inverse = 1.0 / filter_scale;
    std::vector<Taps> result;
    result.reserve(static_cast<std::size_t>(output));
    for (int32_t index = 0; index < output; ++index) {
        const double center = (index + 0.5) * scale;
        const auto begin = static_cast<int32_t>(std::max(0.0, center - support + 0.5));
        const auto end =
            static_cast<int32_t>(std::min(static_cast<double>(input), center + support + 0.5));
        std::vector<double> values(static_cast<std::size_t>(end - begin));
        double sum = 0.0;
        for (int32_t source = begin; source < end; ++source) {
            const double weight = lanczos((source - center + 0.5) * inverse);
            values[static_cast<std::size_t>(source - begin)] = weight;
            sum += weight;
        }
        Taps taps{begin, {}};
        taps.weights.reserve(values.size());
        for (const double value : values) {
            const double fixed = (sum != 0.0 ? value / sum : value) * kUnit;
            taps.weights.push_back(static_cast<int32_t>(fixed < 0.0 ? fixed - 0.5 : fixed + 0.5));
        }
        result.push_back(std::move(taps));
    }
    return result;
}

uint8_t rounded_pixel(int64_t sum) {
    // The caller has already added the half-unit rounding bias. Division and
    // clamping avoid an implementation-defined signed negative shift.
    return static_cast<uint8_t>(std::clamp(sum / kUnit, int64_t{0}, int64_t{255}));
}

} // namespace

ReferenceImageShape reference_image_shape(int32_t height, int32_t width) {
    (void)image_bytes(height, width);
    if (static_cast<int64_t>(height) > int64_t{4} * width ||
        static_cast<int64_t>(width) > int64_t{4} * height)
        throw std::invalid_argument("MiniMax-H3 image-reference aspect must be within 1:4 and 4:1");
    const double scale = 2048.0 / std::min(height, width);
    return {aligned_even_round(height * scale / 32), aligned_even_round(width * scale / 32)};
}

std::vector<uint8_t> resize_reference_rgb_lanczos(const uint8_t* rgb, std::size_t bytes,
                                                  int32_t height, int32_t width,
                                                  ReferenceImageShape target) {
    const auto expected = image_bytes(height, width);
    const auto size = image_bytes(target.height, target.width);
    if (!rgb || bytes != expected)
        throw std::invalid_argument("MiniMax-H3 reference RGB payload size mismatch");
    if (height == target.height && width == target.width)
        return {rgb, rgb + bytes};
    const bool horizontal = width != target.width;
    const auto x_taps = horizontal ? coefficients(width, target.width) : std::vector<Taps>{};
    const auto row_bytes = static_cast<std::size_t>(target.width) * 3;
    auto resample_row = [&](int32_t row) {
        const auto* source = rgb + static_cast<std::size_t>(row) * width * 3;
        if (!horizontal)
            return std::vector<uint8_t>(source, source + row_bytes);
        std::vector<uint8_t> result(row_bytes);
        for (int32_t x = 0; x < target.width; ++x) {
            const auto& taps = x_taps[static_cast<std::size_t>(x)];
            for (int32_t channel = 0; channel < 3; ++channel) {
                int64_t sum = kUnit / 2;
                for (std::size_t i = 0; i < taps.weights.size(); ++i)
                    sum += static_cast<int64_t>(source[(taps.start + i) * 3 + channel]) *
                           taps.weights[i];
                result[static_cast<std::size_t>(x) * 3 + channel] = rounded_pixel(sum);
            }
        }
        return result;
    };
    std::vector<uint8_t> result(size);
    if (height == target.height) {
        for (int32_t y = 0; y < height; ++y) {
            const auto row = resample_row(y);
            std::copy(row.begin(), row.end(), result.begin() + y * row_bytes);
        }
        return result;
    }
    const auto y_taps = coefficients(height, target.height);
    std::deque<std::vector<uint8_t>> rows;
    int32_t first = 0;
    int32_t next = 0;
    for (int32_t y = 0; y < target.height; ++y) {
        const auto& taps = y_taps[static_cast<std::size_t>(y)];
        while (first < taps.start && !rows.empty()) {
            rows.pop_front();
            ++first;
        }
        if (rows.empty())
            first = next = taps.start;
        const auto end = taps.start + static_cast<int32_t>(taps.weights.size());
        for (; next < end; ++next)
            rows.push_back(resample_row(next));
        for (std::size_t x = 0; x < row_bytes; ++x) {
            int64_t sum = kUnit / 2;
            for (std::size_t i = 0; i < taps.weights.size(); ++i)
                sum += static_cast<int64_t>(rows[i][x]) * taps.weights[i];
            result[static_cast<std::size_t>(y) * row_bytes + x] = rounded_pixel(sum);
        }
    }
    return result;
}

PreparedReferenceImage prepare_reference_image(const uint8_t* rgb, std::size_t bytes,
                                               int32_t height, int32_t width) {
    const auto shape = reference_image_shape(height, width);
    return {shape, resize_reference_rgb_lanczos(rgb, bytes, height, width, shape)};
}

std::vector<float> pack_visual_reference_rgb_tile(const PreparedReferenceImage& image, int32_t top,
                                                  int32_t left, int32_t tile_height,
                                                  int32_t tile_width) {
    const auto height = image.shape.height;
    const auto width = image.shape.width;
    if (image.rgb.size() != image_bytes(height, width) || top < 0 || left < 0 || tile_height < 32 ||
        tile_height > 256 || tile_width < 32 || tile_width > 256 || top % 16 || left % 16 ||
        tile_height % 16 || tile_width % 16 || static_cast<int64_t>(top) + tile_height > height ||
        static_cast<int64_t>(left) + tile_width > width)
        throw std::invalid_argument("MiniMax-H3 VisualVAE image tile is outside aligned RGB input");
    constexpr float mean[] = {0.485F, 0.456F, 0.406F};
    constexpr float std[] = {0.229F, 0.224F, 0.225F};
    const auto plane = static_cast<std::size_t>(tile_height) * tile_width;
    std::vector<float> pixels(3 * plane);
    for (int32_t c = 0; c < 3; ++c)
        for (int32_t y = 0; y < tile_height; ++y)
            for (int32_t x = 0; x < tile_width; ++x) {
                const auto index = (static_cast<std::size_t>(top + y) * width + left + x) * 3 + c;
                const float unit = static_cast<float>(image.rgb[index]) / 255.0F;
                pixels[static_cast<std::size_t>(c) * plane + y * tile_width + x] =
                    (unit - mean[c]) / std[c];
            }
    return pixels;
}

} // namespace trtmc::minimax_h3
