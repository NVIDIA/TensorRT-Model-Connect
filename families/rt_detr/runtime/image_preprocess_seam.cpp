/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rt_detr/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace {

struct Filter {
    int32_t start;
    std::vector<int32_t> coefficients;
};

std::vector<Filter> bilinear_filters(int32_t source, int32_t target) {
    const double scale = static_cast<double>(source) / target;
    const double stretch = std::max(1.0, scale);
    std::vector<Filter> result;
    result.reserve(target);
    for (int32_t output = 0; output < target; ++output) {
        const double center = (output + 0.5) * scale;
        const auto start = std::max(0, static_cast<int32_t>(center - stretch + 0.5));
        const auto end = std::min(source, static_cast<int32_t>(center + stretch + 0.5));
        std::vector<double> weights;
        double sum = 0;
        for (auto input = start; input < end; ++input) {
            const double weight = std::max(0.0, 1.0 - std::abs((input - center + 0.5) / stretch));
            weights.push_back(weight);
            sum += weight;
        }
        Filter filter{start, {}};
        for (const auto weight : weights)
            filter.coefficients.push_back(
                static_cast<int32_t>(std::lround(weight / sum * (1 << 22))));
        result.push_back(std::move(filter));
    }
    return result;
}

} // namespace

std::vector<float> preprocess_rt_detr_image(const float* pixels, int32_t image_height,
                                            int32_t image_width,
                                            const RtDetrPreprocessConfig& config) {
    if (pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("rt_detr preprocessing received an empty image");
    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;
    if (out_h <= 0 || out_w <= 0)
        throw std::invalid_argument("rt_detr preprocessing target size must be positive");

    // Pillow's RGB bilinear resize widens the filter for downsampling and
    // rounds to bytes after each separable pass, with fixed 22-bit weights.
    const auto count = static_cast<std::size_t>(image_height) * image_width * 3;
    std::vector<uint8_t> source(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 1)
            throw std::invalid_argument("rt_detr float RGB pixels must be finite in [0,1]");
        source[i] = static_cast<uint8_t>(std::lround(pixels[i] * 255.0F));
    }
    const auto horizontal = bilinear_filters(image_width, out_w);
    std::vector<uint8_t> intermediate(static_cast<std::size_t>(image_height) * out_w * 3);
    for (int32_t row = 0; row < image_height; ++row)
        for (int32_t column = 0; column < out_w; ++column)
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto& filter = horizontal[column];
                int64_t value = 1 << 21;
                for (std::size_t i = 0; i < filter.coefficients.size(); ++i)
                    value +=
                        static_cast<int64_t>(filter.coefficients[i]) *
                        source[(static_cast<std::size_t>(row) * image_width + filter.start + i) *
                                   3 +
                               channel];
                intermediate[(static_cast<std::size_t>(row) * out_w + column) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<int64_t>(value >> 22, 0, 255));
            }
    const auto vertical = bilinear_filters(image_height, out_h);
    std::vector<float> out(static_cast<std::size_t>(3) * out_h * out_w);
    for (int32_t row = 0; row < out_h; ++row)
        for (int32_t column = 0; column < out_w; ++column)
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto& filter = vertical[row];
                int64_t value = 1 << 21;
                for (std::size_t i = 0; i < filter.coefficients.size(); ++i)
                    value += static_cast<int64_t>(filter.coefficients[i]) *
                             intermediate[((static_cast<std::size_t>(filter.start) + i) * out_w +
                                           column) *
                                              3 +
                                          channel];
                out[(static_cast<std::size_t>(channel) * out_h + row) * out_w + column] =
                    static_cast<float>(std::clamp<int64_t>(value >> 22, 0, 255)) / 255.0F;
            }
    return out;
}

} // namespace trtmc
