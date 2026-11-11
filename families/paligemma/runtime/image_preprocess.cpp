/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/paligemma/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::paligemma {
namespace {

constexpr std::int32_t kPrecisionBits = 22;
constexpr std::int64_t kScale = std::int64_t{1} << kPrecisionBits;
constexpr std::int64_t kRounding = std::int64_t{1} << (kPrecisionBits - 1);

struct Span {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<Span> plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<Span> spans(static_cast<std::size_t>(output_size));
    for (std::int32_t index = 0; index < output_size; ++index) {
        const double center = (static_cast<double>(index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("PaliGemma resize produced an empty filter support");
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t source = first; source < end; ++source) {
            const double weight =
                cubic((static_cast<double>(source) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(source - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("PaliGemma resize has invalid filter coefficients");
        auto& span = spans[static_cast<std::size_t>(index)];
        span.first = first;
        for (const double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kScale);
            if (!std::isfinite(scaled) ||
                std::fabs(scaled) > static_cast<double>(std::numeric_limits<std::int32_t>::max()))
                throw std::runtime_error("PaliGemma resize coefficient overflowed");
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return spans;
}

std::uint8_t apply(const std::uint8_t* source, std::int32_t stride, const Span& span) {
    std::int64_t sum = kRounding;
    for (std::size_t i = 0; i < span.weights.size(); ++i)
        sum += static_cast<std::int64_t>(
                   source[static_cast<std::ptrdiff_t>(span.first + static_cast<std::int32_t>(i)) *
                          stride]) *
               span.weights[i];
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPrecisionBits, 255));
}

} // namespace

std::vector<std::uint8_t> resize_pillow_bicubic(const std::uint8_t* rgb, std::int32_t height,
                                                std::int32_t width, std::int32_t output_height,
                                                std::int32_t output_width) {
    if (rgb == nullptr || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0)
        throw std::invalid_argument("PaliGemma resize requires non-empty images");
    // Pillow skips a pass whose size does not change.
    std::vector<std::uint8_t> horizontal;
    const std::uint8_t* rows = rgb;
    if (output_width != width) {
        const auto spans = plan(width, output_width);
        horizontal.resize(static_cast<std::size_t>(height) * output_width * 3U);
        for (std::int32_t y = 0; y < height; ++y)
            for (std::int32_t x = 0; x < output_width; ++x)
                for (std::int32_t c = 0; c < 3; ++c)
                    horizontal[(static_cast<std::size_t>(y) * output_width + x) * 3U + c] =
                        apply(rgb + static_cast<std::size_t>(y) * width * 3U + c, 3,
                              spans[static_cast<std::size_t>(x)]);
        rows = horizontal.data();
    }
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_height) * output_width * 3U);
    if (output_height == height) {
        std::copy_n(rows, output.size(), output.begin());
        return output;
    }
    const auto spans = plan(height, output_height);
    for (std::int32_t y = 0; y < output_height; ++y)
        for (std::int32_t x = 0; x < output_width; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
                output[(static_cast<std::size_t>(y) * output_width + x) * 3U + c] =
                    apply(rows + static_cast<std::size_t>(x) * 3U + c, output_width * 3,
                          spans[static_cast<std::size_t>(y)]);
    return output;
}

std::vector<float> preprocess_image(const std::uint8_t* rgb, std::int32_t height,
                                    std::int32_t width, const ImageNormalization& config) {
    if (config.size <= 0 || !(config.rescale_factor > 0.0))
        throw std::invalid_argument("PaliGemma image size and rescale factor must be positive");
    for (const float value : config.std)
        if (value == 0.0F || !std::isfinite(value))
            throw std::invalid_argument("PaliGemma image std must be finite and non-zero");
    const auto resized = resize_pillow_bicubic(rgb, height, width, config.size, config.size);
    const auto plane = static_cast<std::size_t>(config.size) * config.size;
    std::vector<float> pixels(3U * plane);
    for (std::size_t index = 0; index < plane; ++index)
        for (std::size_t c = 0; c < 3; ++c) {
            const auto value = static_cast<float>(static_cast<double>(resized[index * 3U + c]) *
                                                  config.rescale_factor);
            pixels[c * plane + index] = (value - config.mean[c]) / config.std[c];
        }
    return pixels;
}

} // namespace trtmc::paligemma
