/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chinese_clip/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

// Pillow's bicubic resize, reproduced coefficient for coefficient: the
// reference processor resamples with PIL (resample=3), and a float bilinear or
// a Catmull-Rom cubic lands a few least-significant bits away on every pixel,
// which is enough to move the last digits of a cosine.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);

struct PillowSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double pillow_cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<PillowSpan> make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<PillowSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first = static_cast<std::int32_t>(std::max(0.0, center - support + 0.5));
        const auto end = static_cast<std::int32_t>(
            std::min(static_cast<double>(input_size), center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("chinese_clip Pillow resize produced empty support");

        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_cubic((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("chinese_clip Pillow resize has invalid coefficients");
        span.weights.reserve(floating.size());
        for (double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kPillowScale);
            if (!std::isfinite(scaled) ||
                scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("chinese_clip Pillow resize coefficient overflowed");
            }
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

std::uint8_t apply_pillow_span(const std::uint8_t* source, std::size_t stride,
                               const PillowSpan& span) {
    std::int64_t sum = kPillowRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        sum += static_cast<std::int64_t>(
                   source[(static_cast<std::size_t>(span.first) + index) * stride]) *
               span.weights[index];
    }
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPillowPrecisionBits, 255));
}

std::vector<float> resize_pillow_bicubic(const float* source, std::int32_t input_h,
                                         std::int32_t input_w, std::int32_t output_h,
                                         std::int32_t output_w) {
    const auto input_stride = static_cast<std::size_t>(input_w) * 3U;
    if (static_cast<std::size_t>(input_h) >
        std::numeric_limits<std::size_t>::max() / sizeof(float) / input_stride)
        throw std::invalid_argument("chinese_clip source image storage overflows");
    const auto output_stride = static_cast<std::size_t>(output_w) * 3U;
    const auto horizontal_plan = make_pillow_plan(input_w, output_w);
    const auto vertical_plan = make_pillow_plan(input_h, output_h);
    // Preserve Pillow's horizontal byte rounding with four cached rows instead
    // of a source-height by output-width image. Upsampling reuses each row;
    // downsampling can revisit a row only within the fixed filter support.
    std::vector<std::uint8_t> input_row(input_stride);
    std::array<std::vector<std::uint8_t>, 4> filtered_rows;
    for (auto& row : filtered_rows)
        row.resize(output_stride);
    std::array<std::size_t, 4> row_ids;
    row_ids.fill(std::numeric_limits<std::size_t>::max());
    std::vector<std::int64_t> sums(output_stride);
    std::vector<float> output(static_cast<std::size_t>(output_h) * output_stride);
    for (std::int32_t y = 0; y < output_h; ++y) {
        std::fill(sums.begin(), sums.end(), kPillowRounding);
        const auto& vertical = vertical_plan[static_cast<std::size_t>(y)];
        for (std::size_t row = 0; row < vertical.weights.size(); ++row) {
            const auto source_y = static_cast<std::size_t>(vertical.first) + row;
            const auto slot = source_y % filtered_rows.size();
            auto& filtered = filtered_rows[slot];
            if (row_ids[slot] != source_y) {
                const auto* first = source + source_y * input_stride;
                std::transform(first, first + input_stride, input_row.begin(), [](float value) {
                    return static_cast<std::uint8_t>(
                        std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
                });
                for (std::int32_t x = 0; x < output_w; ++x) {
                    const auto& horizontal = horizontal_plan[static_cast<std::size_t>(x)];
                    for (std::int32_t channel = 0; channel < 3; ++channel)
                        filtered[static_cast<std::size_t>(x) * 3U + channel] =
                            apply_pillow_span(input_row.data() + channel, 3, horizontal);
                }
                row_ids[slot] = source_y;
            }
            for (std::size_t index = 0; index < output_stride; ++index)
                sums[index] += static_cast<std::int64_t>(filtered[index]) * vertical.weights[row];
        }
        for (std::size_t index = 0; index < output_stride; ++index) {
            const auto sum = sums[index];
            const auto pixel =
                sum <= 0 ? 0 : std::min<std::int64_t>(sum >> kPillowPrecisionBits, 255);
            output[static_cast<std::size_t>(y) * output_stride + index] =
                static_cast<float>(pixel) / 255.0F;
        }
    }
    return output;
}

void validate(const ChineseClipPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("chinese_clip input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("chinese_clip image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("chinese_clip image std must be non-zero");
    }
}

} // namespace

ChineseClipResizeShape
compute_chinese_clip_resize_shape(std::int32_t image_height, std::int32_t image_width,
                                  const ChineseClipPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("chinese_clip source dimensions must be positive");
    validate(config);
    if (config.input_image_h != config.input_image_w)
        throw std::invalid_argument("chinese_clip crops to a square, so both sides must match");

    // The published Chinese-CLIP processor explicitly resizes to height/width
    // with do_center_crop=false. Both source axes therefore span the output.
    return {config.input_image_h, config.input_image_w};
}

std::vector<float> preprocess_chinese_clip_image(const float* image_pixels,
                                                 std::int32_t image_height,
                                                 std::int32_t image_width,
                                                 const ChineseClipPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("chinese_clip source image must be non-empty");
    validate(config);

    const auto shape = compute_chinese_clip_resize_shape(image_height, image_width, config);
    const auto resized =
        resize_pillow_bicubic(image_pixels, image_height, image_width, shape.height, shape.width);

    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::int32_t y = 0; y < config.input_image_h; ++y) {
        for (std::int32_t x = 0; x < config.input_image_w; ++x) {
            const auto source = (static_cast<std::size_t>(y) * shape.width + x) * 3U;
            for (std::int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                pixel_values[channel * plane + static_cast<std::size_t>(y) * config.input_image_w +
                             x] = (resized[source + channel] - config.image_mean[channel]) /
                                  config.image_std[channel];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
