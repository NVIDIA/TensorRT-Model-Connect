/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::lighton_ocr {
namespace {

struct Filter {
    std::int32_t start;
    std::vector<std::int32_t> coefficients;
};

double cubic(double value) {
    value = std::abs(value);
    if (value < 1)
        return ((1.5 * value - 2.5) * value) * value + 1;
    return value < 2 ? ((-0.5 * value + 2.5) * value - 4) * value + 2 : 0;
}

std::vector<Filter> filters(std::int32_t source, std::int32_t target) {
    const double scale = static_cast<double>(source) / target;
    const double stretch = std::max(1.0, scale);
    std::vector<Filter> result;
    for (std::int32_t output = 0; output < target; ++output) {
        const double center = (output + 0.5) * scale;
        const auto start = std::max(0, static_cast<std::int32_t>(center - 2 * stretch + 0.5));
        const auto end = std::min(source, static_cast<std::int32_t>(center + 2 * stretch + 0.5));
        std::vector<double> weights;
        double sum = 0;
        for (auto input = start; input < end; ++input) {
            const double weight = cubic((input - center + 0.5) / stretch);
            weights.push_back(weight);
            sum += weight;
        }
        Filter filter{start, {}};
        for (const double weight : weights)
            filter.coefficients.push_back(
                static_cast<std::int32_t>(std::lround(weight / sum * (1 << 22))));
        result.push_back(std::move(filter));
    }
    return result;
}

// Pillow RGB bicubic uses separable antialiasing with fixed 22-bit weights,
// half-pixel centers, and byte rounding/clamping after each axis.
std::vector<std::uint8_t> resize(const std::vector<std::uint8_t>& source, std::int32_t width,
                                 std::int32_t height, std::int32_t output_width,
                                 std::int32_t output_height) {
    const auto horizontal = filters(width, output_width);
    std::vector<std::uint8_t> intermediate(static_cast<std::size_t>(height) * output_width * 3);
    for (std::int32_t y = 0; y < height; ++y)
        for (std::int32_t x = 0; x < output_width; ++x)
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto& filter = horizontal[x];
                std::int64_t value = 1 << 21;
                for (std::size_t i = 0; i < filter.coefficients.size(); ++i)
                    value += static_cast<std::int64_t>(filter.coefficients[i]) *
                             source[(static_cast<std::size_t>(y) * width + filter.start + i) * 3 +
                                    channel];
                intermediate[(static_cast<std::size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<std::uint8_t>(std::clamp<std::int64_t>(value >> 22, 0, 255));
            }
    const auto vertical = filters(height, output_height);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_height) * output_width * 3);
    for (std::int32_t y = 0; y < output_height; ++y)
        for (std::int32_t x = 0; x < output_width; ++x)
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto& filter = vertical[y];
                std::int64_t value = 1 << 21;
                for (std::size_t i = 0; i < filter.coefficients.size(); ++i)
                    value +=
                        static_cast<std::int64_t>(filter.coefficients[i]) *
                        intermediate[((static_cast<std::size_t>(filter.start) + i) * output_width +
                                      x) *
                                         3 +
                                     channel];
                output[(static_cast<std::size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<std::uint8_t>(std::clamp<std::int64_t>(value >> 22, 0, 255));
            }
    return output;
}

std::vector<std::uint8_t> image_bytes(const internal::ImageView& image) {
    if (!image.data || !image.height || !image.width || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument("LightOnOCR requires one nonempty host RGB image");
    const std::uint64_t count = static_cast<std::uint64_t>(image.height) * image.width * 3;
    const std::size_t item_size = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / item_size ||
        image.byte_size != count * item_size)
        throw std::invalid_argument("LightOnOCR RGB storage does not match its dimensions");
    std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
    if (image.format == internal::ImageFormat::UInt8) {
        std::memcpy(result.data(), image.data, result.size());
    } else {
        const auto* bytes = static_cast<const unsigned char*>(image.data);
        for (std::size_t i = 0; i < result.size(); ++i) {
            float value;
            std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("LightOnOCR float RGB pixels must be finite in [0,1]");
            result[i] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
        }
    }
    return result;
}

} // namespace

PreparedImage preprocess(const internal::ImageView& image, const ImageConfig& config) {
    if (config.patch_size <= 0 || config.spatial_merge_size <= 0 ||
        config.max_image_size < config.patch_size * config.spatial_merge_size ||
        config.max_image_size % (config.patch_size * config.spatial_merge_size) != 0 ||
        !std::isfinite(config.rescale_factor) || config.rescale_factor <= 0)
        throw std::invalid_argument("LightOnOCR invalid image processor geometry");
    for (std::size_t channel = 0; channel < 3; ++channel)
        if (!std::isfinite(config.mean[channel]) ||
            !std::isfinite(config.standard_deviation[channel]) ||
            config.standard_deviation[channel] <= 0)
            throw std::invalid_argument("LightOnOCR invalid image normalization");
    const auto source = image_bytes(image);
    const auto multiple = config.patch_size;
    const double ratio = std::max(static_cast<double>(image.height) / config.max_image_size,
                                  static_cast<double>(image.width) / config.max_image_size);
    auto height = static_cast<std::int32_t>(image.height);
    auto width = static_cast<std::int32_t>(image.width);
    if (ratio > 1) {
        height = static_cast<std::int32_t>(std::floor(height / ratio));
        width = static_cast<std::int32_t>(std::floor(width / ratio));
    }
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("LightOnOCR resized image would have an empty dimension");
    height = ((height - 1) / multiple + 1) * multiple;
    width = ((width - 1) / multiple + 1) * multiple;
    if (height < config.patch_size * config.spatial_merge_size ||
        width < config.patch_size * config.spatial_merge_size)
        throw std::invalid_argument(
            "LightOnOCR image must contain a merged patch in each dimension");
    const auto bytes = resize(source, static_cast<std::int32_t>(image.width),
                              static_cast<std::int32_t>(image.height), width, height);
    PreparedImage output{height, width, {}, {}};
    const auto area = static_cast<std::size_t>(height) * width;
    output.pixels.resize(area * 3);
    for (std::size_t channel = 0; channel < 3; ++channel)
        for (std::size_t pixel = 0; pixel < area; ++pixel) {
            const float scaled =
                static_cast<float>(static_cast<double>(bytes[pixel * 3 + channel]) *
                                   static_cast<double>(config.rescale_factor));
            output.pixels[channel * area + pixel] =
                (scaled - config.mean[channel]) / config.standard_deviation[channel];
        }
    const auto stride = config.max_image_size / config.patch_size;
    for (std::int32_t y = 0; y < height / config.patch_size; ++y)
        for (std::int32_t x = 0; x < width / config.patch_size; ++x)
            output.position_ids.push_back(y * stride + x);
    return output;
}

} // namespace trtmc::lighton_ocr
