/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/trocr/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::trocr {
namespace {

// Pillow's resampler, coefficient for coefficient: the reference processor
// resizes with PIL, and a float resize lands a few levels away on edges.
constexpr std::int32_t kPrecisionBits = 22;
constexpr double kScale = static_cast<double>(std::int64_t{1} << kPrecisionBits);
constexpr std::int64_t kRounding = std::int64_t{1} << (kPrecisionBits - 1);

struct Span {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double bilinear(double value) {
    value = std::abs(value);
    return value < 1.0 ? 1.0 - value : 0.0;
}

std::vector<Span> plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = filter_scale; // bilinear support is 1
    // Pillow multiplies by the reciprocal; dividing can differ in the last bit.
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<Span> spans(static_cast<std::size_t>(output_size));
    for (std::int32_t out = 0; out < output_size; ++out) {
        const double center = (out + 0.5) * scale;
        const auto first = std::max(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end = std::min(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("trocr resize produced an empty filter support");
        std::vector<double> weights(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight = bilinear((index - center + 0.5) * inverse_filter_scale);
            weights[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        auto& span = spans[static_cast<std::size_t>(out)];
        span.first = first;
        span.weights.reserve(weights.size());
        for (double weight : weights) {
            const double fixed = (total != 0.0 ? weight / total : weight) * kScale;
            span.weights.push_back(
                static_cast<std::int32_t>(fixed < 0.0 ? fixed - 0.5 : fixed + 0.5));
        }
    }
    return spans;
}

std::uint8_t apply(const std::uint8_t* source, std::size_t stride, const Span& span) {
    std::int64_t sum = kRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index)
        sum += static_cast<std::int64_t>(
                   source[(static_cast<std::size_t>(span.first) + index) * stride]) *
               span.weights[index];
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPrecisionBits, 255));
}

std::vector<std::uint8_t> horizontal_pass(const std::uint8_t* rgb, std::int32_t height,
                                          std::int32_t width, std::int32_t output_width) {
    const auto spans = plan(width, output_width);
    const auto rows = static_cast<std::size_t>(height);
    const auto columns = static_cast<std::size_t>(output_width);
    std::vector<std::uint8_t> output(rows * columns * 3);
    for (std::size_t y = 0; y < rows; ++y) {
        const auto* row = rgb + y * static_cast<std::size_t>(width) * 3;
        for (std::size_t x = 0; x < columns; ++x) {
            for (std::size_t c = 0; c < 3; ++c)
                output[(y * columns + x) * 3 + c] = apply(row + c, 3, spans[x]);
        }
    }
    return output;
}

std::vector<std::uint8_t> vertical_pass(const std::uint8_t* rgb, std::int32_t height,
                                        std::int32_t width, std::int32_t output_height) {
    const auto spans = plan(height, output_height);
    const auto rows = static_cast<std::size_t>(output_height);
    const auto columns = static_cast<std::size_t>(width);
    std::vector<std::uint8_t> output(rows * columns * 3);
    for (std::size_t y = 0; y < rows; ++y) {
        for (std::size_t x = 0; x < columns; ++x) {
            for (std::size_t c = 0; c < 3; ++c)
                output[(y * columns + x) * 3 + c] = apply(rgb + x * 3 + c, columns * 3, spans[y]);
        }
    }
    return output;
}

std::size_t checked_pixels(std::uint32_t height, std::uint32_t width) {
    constexpr auto kLimit = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (height == 0 || width == 0 || height > kLimit || width > kLimit ||
        static_cast<std::uint64_t>(height) >
            std::numeric_limits<std::size_t>::max() / width / 4 / sizeof(float))
        throw std::invalid_argument("trocr image dimensions are empty or too large");
    return static_cast<std::size_t>(height) * width;
}

} // namespace

std::vector<std::uint8_t> to_rgb8(const internal::ImageView& image) {
    const auto pixels = checked_pixels(image.height, image.width);
    const auto channels = static_cast<std::size_t>(image.channels);
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::invalid_argument("trocr reads grayscale, RGB or RGBA images");
    const bool is_float = image.format == internal::ImageFormat::Float32;
    if (!is_float && image.format != internal::ImageFormat::UInt8)
        throw std::invalid_argument("trocr reads uint8 or float32 images");
    const auto element = is_float ? sizeof(float) : sizeof(std::uint8_t);
    if (image.data == nullptr || image.byte_size != pixels * channels * element)
        throw std::invalid_argument("trocr image data does not match its dimensions");

    const auto level = [&](std::size_t index) -> std::uint8_t {
        if (!is_float)
            return static_cast<const std::uint8_t*>(image.data)[index];
        const float value = static_cast<const float*>(image.data)[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("trocr float32 pixels must be finite and in [0, 1]");
        return static_cast<std::uint8_t>(std::lround(value * 255.0F));
    };
    std::vector<std::uint8_t> rgb(pixels * 3);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (std::size_t c = 0; c < 3; ++c)
            rgb[pixel * 3 + c] = level(pixel * channels + (channels == 1 ? 0 : c));
    }
    return rgb;
}

std::vector<std::uint8_t> resize_bilinear(const std::uint8_t* rgb, std::int32_t height,
                                          std::int32_t width, std::int32_t output_height,
                                          std::int32_t output_width) {
    if (rgb == nullptr || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0)
        throw std::invalid_argument("trocr resize needs a non-empty source and target");
    // Pillow resizes vertically first when the height shrinks by more than
    // twice as many rows as the width shrinks by columns. Each pass rounds to
    // 8 bits, so the order changes the result.
    const auto rows_removed = static_cast<std::int64_t>(height) - output_height;
    const auto columns_removed = static_cast<std::int64_t>(width) - output_width;
    if (rows_removed > 0 && rows_removed > 2 * columns_removed) {
        const auto vertical = vertical_pass(rgb, height, width, output_height);
        return horizontal_pass(vertical.data(), output_height, width, output_width);
    }
    const auto horizontal = horizontal_pass(rgb, height, width, output_width);
    return vertical_pass(horizontal.data(), height, output_width, output_height);
}

std::vector<float> preprocess(const internal::ImageView& image, const PreprocessConfig& config) {
    if (config.height <= 0 || config.width <= 0 || !(config.rescale_factor > 0.0))
        throw std::invalid_argument("trocr preprocessing configuration is invalid");
    const auto rgb = to_rgb8(image);
    const auto resized =
        resize_bilinear(rgb.data(), static_cast<std::int32_t>(image.height),
                        static_cast<std::int32_t>(image.width), config.height, config.width);
    const auto plane = static_cast<std::size_t>(config.height) * config.width;
    std::vector<float> pixel_values(3 * plane);
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        for (std::size_t c = 0; c < 3; ++c) {
            // numpy: image.astype(float64) * scale, then astype(float32);
            // normalize then runs in float32.
            const auto rescaled = static_cast<float>(static_cast<double>(resized[pixel * 3 + c]) *
                                                     config.rescale_factor);
            pixel_values[c * plane + pixel] =
                (rescaled - config.image_mean[c]) / config.image_std[c];
        }
    }
    return pixel_values;
}

} // namespace trtmc::trocr
