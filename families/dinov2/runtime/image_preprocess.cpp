/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/image_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace {

// Pillow's fixed-point separable resampler, as used by the slow BitImageProcessor.
// Pillow accumulates 8-bit resampling in 32-bit integers; so does this code.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr double kPillowScale = static_cast<double>(std::int64_t{1} << kPillowPrecisionBits);
constexpr std::int32_t kPillowRounding = std::int32_t{1} << (kPillowPrecisionBits - 1);

// Coefficients for a contiguous window of output indices, each with Pillow's own span.
struct PillowPlan {
    std::vector<std::int32_t> first;
    std::vector<std::int32_t> count;
    std::vector<std::int32_t> weights; // row-major, `taps` entries per output index
    std::int32_t taps{0};
};

// Pillow's bicubic_filter, including its evaluation order.
double pillow_cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return (((value - 5.0) * value + 8.0) * value - 4.0) * kA;
    return 0.0;
}

PillowPlan make_pillow_plan(std::int32_t input_size, std::int32_t output_size,
                            std::int32_t window_begin, std::int32_t window_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    PillowPlan plan;
    plan.taps = static_cast<std::int32_t>(std::ceil(support)) * 2 + 1;
    plan.first.resize(static_cast<std::size_t>(window_size));
    plan.count.resize(static_cast<std::size_t>(window_size));
    plan.weights.assign(static_cast<std::size_t>(window_size) * plan.taps, 0);
    std::vector<double> floating(static_cast<std::size_t>(plan.taps));
    for (std::int32_t index = 0; index < window_size; ++index) {
        const double center = (static_cast<double>(window_begin + index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first || end - first > plan.taps)
            throw std::runtime_error("DINOv2 Pillow resize produced an invalid support");
        double total = 0.0;
        for (std::int32_t tap = 0; tap < end - first; ++tap) {
            floating[static_cast<std::size_t>(tap)] = pillow_cubic(
                (static_cast<double>(first + tap) - center + 0.5) * inverse_filter_scale);
            total += floating[static_cast<std::size_t>(tap)];
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("DINOv2 Pillow resize has invalid coefficients");
        plan.first[static_cast<std::size_t>(index)] = first;
        plan.count[static_cast<std::size_t>(index)] = end - first;
        auto* weights = plan.weights.data() + static_cast<std::size_t>(index) * plan.taps;
        for (std::int32_t tap = 0; tap < end - first; ++tap) {
            const double scaled = floating[static_cast<std::size_t>(tap)] / total * kPillowScale;
            weights[tap] = static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
        }
    }
    return plan;
}

std::uint8_t pillow_clip(std::int32_t sum) {
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min(sum >> kPillowPrecisionBits, 255));
}

void validate_config(const Dinov2PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0 ||
        config.resize_shortest_edge < config.input_image_h ||
        config.resize_shortest_edge < config.input_image_w)
        throw std::invalid_argument("DINOv2 crop must fit inside the resized shortest edge");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("DINOv2 image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (!std::isfinite(value) || value <= 0.0F)
            throw std::invalid_argument("DINOv2 image std must be finite and positive");
    }
}

} // namespace

Dinov2ImageGeometry compute_dinov2_image_geometry(int32_t image_height, int32_t image_width,
                                                  const Dinov2PreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("DINOv2 source image must be non-empty");
    validate_config(config);
    // transformers.image_transforms.get_resize_output_image_size(default_to_square=False):
    // the short edge becomes `shortest_edge`, the long edge int(shortest_edge * long / short).
    const auto shortest = static_cast<std::int64_t>(config.resize_shortest_edge);
    Dinov2ImageGeometry geometry;
    if (image_width <= image_height) {
        geometry.resized_w = config.resize_shortest_edge;
        geometry.resized_h = static_cast<int32_t>(shortest * image_height / image_width);
    } else {
        geometry.resized_h = config.resize_shortest_edge;
        geometry.resized_w = static_cast<int32_t>(shortest * image_width / image_height);
    }
    // transformers.image_transforms.center_crop floors the offset.
    geometry.crop_y = (geometry.resized_h - config.input_image_h) / 2;
    geometry.crop_x = (geometry.resized_w - config.input_image_w) / 2;
    return geometry;
}

std::vector<float> preprocess_dinov2_image(const uint8_t* rgb, int32_t image_height,
                                           int32_t image_width,
                                           const Dinov2PreprocessConfig& config) {
    if (rgb == nullptr)
        throw std::invalid_argument("DINOv2 source image must be non-empty");
    const auto geometry = compute_dinov2_image_geometry(image_height, image_width, config);
    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;

    // Every resized pixel depends only on its own Pillow span, so resampling just the
    // center crop yields the same bytes as resizing the whole image and cropping it.
    // As in Pillow, a pass whose size is unchanged is skipped rather than re-rounded.
    const bool resize_rows = image_height != geometry.resized_h;
    const bool resize_columns = image_width != geometry.resized_w;
    PillowPlan rows;
    int32_t band_begin = geometry.crop_y;
    int32_t band_end = geometry.crop_y + out_h;
    if (resize_rows) {
        rows = make_pillow_plan(image_height, geometry.resized_h, geometry.crop_y, out_h);
        band_begin = rows.first.front();
        band_end = rows.first.back() + rows.count.back();
    }

    // Horizontal pass over the source rows the vertical pass reads.
    const auto band_stride = static_cast<std::size_t>(out_w) * 3U;
    std::vector<std::uint8_t> band(static_cast<std::size_t>(band_end - band_begin) * band_stride);
    const auto source_stride = static_cast<std::size_t>(image_width) * 3U;
    if (resize_columns) {
        const auto columns =
            make_pillow_plan(image_width, geometry.resized_w, geometry.crop_x, out_w);
        for (int32_t y = band_begin; y < band_end; ++y) {
            const auto* source = rgb + static_cast<std::size_t>(y) * source_stride;
            auto* target = band.data() + static_cast<std::size_t>(y - band_begin) * band_stride;
            for (int32_t x = 0; x < out_w; ++x) {
                const auto* weights =
                    columns.weights.data() + static_cast<std::size_t>(x) * columns.taps;
                const auto* pixel =
                    source +
                    static_cast<std::size_t>(columns.first[static_cast<std::size_t>(x)]) * 3U;
                std::int32_t red = kPillowRounding;
                std::int32_t green = kPillowRounding;
                std::int32_t blue = kPillowRounding;
                for (int32_t tap = 0; tap < columns.count[static_cast<std::size_t>(x)]; ++tap) {
                    red += pixel[3 * tap] * weights[tap];
                    green += pixel[3 * tap + 1] * weights[tap];
                    blue += pixel[3 * tap + 2] * weights[tap];
                }
                target[3 * x] = pillow_clip(red);
                target[3 * x + 1] = pillow_clip(green);
                target[3 * x + 2] = pillow_clip(blue);
            }
        }
    } else {
        for (int32_t y = band_begin; y < band_end; ++y) {
            const auto* source = rgb + static_cast<std::size_t>(y) * source_stride +
                                 static_cast<std::size_t>(geometry.crop_x) * 3U;
            std::copy_n(source, band_stride,
                        band.data() + static_cast<std::size_t>(y - band_begin) * band_stride);
        }
    }

    // rescale and normalize are a fixed map from each 8-bit value.
    std::array<std::array<float, 256>, 3> normalized{};
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t v = 0; v < 256; ++v) {
            const float value = static_cast<float>(v) / 255.0F;
            normalized[c][v] = (value - config.image_mean[c]) / config.image_std[c];
        }
    }

    // Vertical pass and NCHW packing.
    const auto plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> pixel_values(3U * plane);
    std::vector<std::int32_t> sums(band_stride);
    std::vector<std::uint8_t> resampled(band_stride);
    for (int32_t y = 0; y < out_h; ++y) {
        const auto output_row = static_cast<std::size_t>(y) * out_w;
        const std::uint8_t* row = nullptr;
        if (resize_rows) {
            const auto* weights = rows.weights.data() + static_cast<std::size_t>(y) * rows.taps;
            const auto first = rows.first[static_cast<std::size_t>(y)] - band_begin;
            std::fill(sums.begin(), sums.end(), kPillowRounding);
            for (int32_t tap = 0; tap < rows.count[static_cast<std::size_t>(y)]; ++tap) {
                const auto* source =
                    band.data() + static_cast<std::size_t>(first + tap) * band_stride;
                for (std::size_t i = 0; i < band_stride; ++i)
                    sums[i] += source[i] * weights[tap];
            }
            for (std::size_t i = 0; i < band_stride; ++i)
                resampled[i] = pillow_clip(sums[i]);
            row = resampled.data();
        } else {
            row = band.data() + static_cast<std::size_t>(y) * band_stride;
        }
        for (int32_t x = 0; x < out_w; ++x) {
            for (std::size_t c = 0; c < 3; ++c)
                pixel_values[c * plane + output_row + static_cast<std::size_t>(x)] =
                    normalized[c][row[3 * static_cast<std::size_t>(x) + c]];
        }
    }
    return pixel_values;
}

} // namespace trtmc
