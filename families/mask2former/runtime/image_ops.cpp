/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mask2former/runtime/image_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::mask2former {
namespace {

// Pillow's 8-bit resampling: coefficients in double, normalized to 22-bit fixed point, accumulated
// in int from a half-unit rounding offset and clipped to [0, 255]. Adapted from Pillow
// src/libImaging/Resample.c (precompute_coeffs, normalize_coeffs_8bpc, ImagingResampleInner); see
// NOTICE.
constexpr int kPrecisionBits = 32 - 8 - 2;

struct Taps {
    std::vector<std::int32_t> first;
    std::vector<std::int32_t> count;
    std::vector<std::int32_t> weights; // [output][max_taps]
    std::int32_t max_taps{0};
};

double bilinear_filter(double x) {
    if (x < 0.0)
        x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

Taps bilinear_taps(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(static_cast<float>(input_size)) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 1.0 * filter_scale;
    Taps taps;
    taps.max_taps = static_cast<std::int32_t>(std::ceil(support)) * 2 + 1;
    taps.first.resize(static_cast<std::size_t>(output_size));
    taps.count.resize(static_cast<std::size_t>(output_size));
    taps.weights.assign(static_cast<std::size_t>(output_size) * taps.max_taps, 0);
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<double> raw(static_cast<std::size_t>(taps.max_taps));
    for (std::int32_t index = 0; index < output_size; ++index) {
        const double center = (index + 0.5) * scale;
        const auto first = std::max(static_cast<std::int32_t>(center - support + 0.5), 0);
        const auto end = std::min(static_cast<std::int32_t>(center + support + 0.5), input_size);
        const auto count = end - first;
        double total = 0.0;
        for (std::int32_t tap = 0; tap < count; ++tap) {
            raw[static_cast<std::size_t>(tap)] =
                bilinear_filter((tap + first - center + 0.5) * inverse_filter_scale);
            total += raw[static_cast<std::size_t>(tap)];
        }
        auto* weights = taps.weights.data() + static_cast<std::size_t>(index) * taps.max_taps;
        for (std::int32_t tap = 0; tap < count; ++tap) {
            const double value = total != 0.0 ? raw[static_cast<std::size_t>(tap)] / total : 0.0;
            weights[tap] = static_cast<std::int32_t>(0.5 + value * (1 << kPrecisionBits));
        }
        taps.first[static_cast<std::size_t>(index)] = first;
        taps.count[static_cast<std::size_t>(index)] = count;
    }
    return taps;
}

std::uint8_t clip8(std::int32_t value) {
    return static_cast<std::uint8_t>(std::clamp(value >> kPrecisionBits, 0, 255));
}

// One pass along rows (horizontal) or columns (vertical) of an HWC RGB image.
std::vector<std::uint8_t> resample_horizontal(const std::vector<std::uint8_t>& input,
                                              std::int32_t height, std::int32_t width,
                                              std::int32_t out_width) {
    const auto taps = bilinear_taps(width, out_width);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(height) * out_width * 3);
    for (std::int32_t y = 0; y < height; ++y) {
        const auto* row = input.data() + static_cast<std::size_t>(y) * width * 3;
        auto* out = output.data() + static_cast<std::size_t>(y) * out_width * 3;
        for (std::int32_t x = 0; x < out_width; ++x) {
            const auto first = taps.first[static_cast<std::size_t>(x)];
            const auto count = taps.count[static_cast<std::size_t>(x)];
            const auto* weights = taps.weights.data() + static_cast<std::size_t>(x) * taps.max_taps;
            for (int channel = 0; channel < 3; ++channel) {
                std::int32_t sum = 1 << (kPrecisionBits - 1);
                for (std::int32_t tap = 0; tap < count; ++tap)
                    sum += row[static_cast<std::size_t>(first + tap) * 3 + channel] * weights[tap];
                out[static_cast<std::size_t>(x) * 3 + channel] = clip8(sum);
            }
        }
    }
    return output;
}

std::vector<std::uint8_t> resample_vertical(const std::vector<std::uint8_t>& input,
                                            std::int32_t height, std::int32_t width,
                                            std::int32_t out_height) {
    const auto taps = bilinear_taps(height, out_height);
    const auto stride = static_cast<std::size_t>(width) * 3;
    std::vector<std::uint8_t> output(static_cast<std::size_t>(out_height) * stride);
    for (std::int32_t y = 0; y < out_height; ++y) {
        const auto first = taps.first[static_cast<std::size_t>(y)];
        const auto count = taps.count[static_cast<std::size_t>(y)];
        const auto* weights = taps.weights.data() + static_cast<std::size_t>(y) * taps.max_taps;
        auto* out = output.data() + static_cast<std::size_t>(y) * stride;
        for (std::size_t column = 0; column < stride; ++column) {
            std::int32_t sum = 1 << (kPrecisionBits - 1);
            for (std::int32_t tap = 0; tap < count; ++tap)
                sum +=
                    input[static_cast<std::size_t>(first + tap) * stride + column] * weights[tap];
            out[column] = clip8(sum);
        }
    }
    return output;
}

// The processor rescales in float64 and casts to float32, then normalizes in float32. The cast is
// defined only for products within float range; normalization_is_finite() checks that first.
float normalize(std::uint8_t value, std::size_t channel, const PreprocessConfig& config) {
    const auto rescaled = static_cast<float>(static_cast<double>(value) * config.rescale_factor);
    return (rescaled - config.mean[channel]) / config.std[channel];
}

void require_extent(std::int32_t height, std::int32_t width, const char* what) {
    if (height <= 0 || width <= 0 ||
        static_cast<std::uint64_t>(height) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(width) / 4)
        throw std::invalid_argument(std::string("Mask2Former ") + what +
                                    " must be a positive size");
}

} // namespace

std::vector<std::uint8_t> resize_bilinear_rgb8(const std::uint8_t* pixels, std::int32_t height,
                                               std::int32_t width, std::int32_t out_height,
                                               std::int32_t out_width) {
    require_extent(height, width, "source image");
    require_extent(out_height, out_width, "resize target");
    if (pixels == nullptr)
        throw std::invalid_argument("Mask2Former source image is null");
    std::vector<std::uint8_t> image(pixels, pixels + static_cast<std::size_t>(height) * width * 3);
    const bool horizontal = out_width != width;
    const bool vertical = out_height != height;
    // Pillow 13 runs the vertical pass first when the height shrinks by more than twice the width's
    // reduction.
    const bool horizontal_first =
        !((height - out_height) > 0 && (height - out_height) > (width - out_width) * 2);
    if (horizontal_first) {
        if (horizontal)
            image = resample_horizontal(image, height, width, out_width);
        if (vertical)
            image = resample_vertical(image, height, out_width, out_height);
    } else {
        if (vertical)
            image = resample_vertical(image, height, width, out_height);
        if (horizontal)
            image = resample_horizontal(image, out_height, width, out_width);
    }
    return image;
}

bool normalization_is_finite(const PreprocessConfig& config) {
    if (!(config.rescale_factor > 0.0) || !std::isfinite(config.rescale_factor) ||
        255.0 * config.rescale_factor > static_cast<double>(std::numeric_limits<float>::max()))
        return false;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        // An infinite std would map every value to a finite zero.
        if (!std::isfinite(config.mean[channel]) || !std::isfinite(config.std[channel]))
            return false;
        for (int value = 0; value <= 255; ++value)
            if (!std::isfinite(normalize(static_cast<std::uint8_t>(value), channel, config)))
                return false;
    }
    return true;
}

std::vector<float> preprocess(const std::uint8_t* pixels, std::int32_t height, std::int32_t width,
                              const PreprocessConfig& config) {
    if (!normalization_is_finite(config))
        throw std::invalid_argument("Mask2Former rescale_factor, image_mean and image_std must "
                                    "normalize 8-bit values to finite float32 values");
    const auto resized = resize_bilinear_rgb8(pixels, height, width, config.height, config.width);
    const auto plane = static_cast<std::size_t>(config.height) * config.width;
    std::vector<float> values(plane * 3);
    for (std::size_t index = 0; index < plane; ++index)
        for (std::size_t channel = 0; channel < 3; ++channel)
            values[channel * plane + index] =
                normalize(resized[index * 3 + channel], channel, config);
    return values;
}

std::vector<std::int32_t> resize_argmax(const float* scores, std::int32_t classes,
                                        std::int32_t score_height, std::int32_t score_width,
                                        std::int32_t height, std::int32_t width) {
    require_extent(score_height, score_width, "score map");
    require_extent(height, width, "segmentation size");
    if (scores == nullptr || classes <= 0)
        throw std::invalid_argument("Mask2Former scores must be nonempty");
    const auto plane = static_cast<std::size_t>(score_height) * score_width;
    if (plane > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(classes))
        throw std::invalid_argument("Mask2Former score maps are too large");
    // Every score, not only those the resize reads: a smaller image samples only some of them.
    if (!std::all_of(scores, scores + plane * static_cast<std::size_t>(classes),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Mask2Former semantic scores are not finite");
    // torch area_pixel_compute_source_index (align_corners=False) in float: src = max(scale * (dst
    // + 0.5) - 0.5, 0), the lower tap is trunc(src) and the upper one is clamped to the last row or
    // column.
    struct Axis {
        std::vector<std::int32_t> low, high;
        std::vector<float> low_weight, high_weight;
    };
    const auto axis = [](std::int32_t input, std::int32_t output) {
        Axis result;
        const float scale = static_cast<float>(input) / static_cast<float>(output);
        for (std::int32_t index = 0; index < output; ++index) {
            const float source = std::max(scale * (static_cast<float>(index) + 0.5F) - 0.5F, 0.0F);
            const auto low = static_cast<std::int32_t>(source);
            const float lambda = source - static_cast<float>(low);
            result.low.push_back(low);
            result.high.push_back(low < input - 1 ? low + 1 : low);
            result.high_weight.push_back(lambda);
            result.low_weight.push_back(1.0F - lambda);
        }
        return result;
    };
    const auto rows = axis(score_height, height);
    const auto columns = axis(score_width, width);
    std::vector<std::int32_t> labels(static_cast<std::size_t>(height) * width, 0);
    std::vector<float> best(static_cast<std::size_t>(width));
    for (std::int32_t y = 0; y < height; ++y) {
        const auto top =
            static_cast<std::size_t>(rows.low[static_cast<std::size_t>(y)]) * score_width;
        const auto bottom =
            static_cast<std::size_t>(rows.high[static_cast<std::size_t>(y)]) * score_width;
        const float top_weight = rows.low_weight[static_cast<std::size_t>(y)];
        const float bottom_weight = rows.high_weight[static_cast<std::size_t>(y)];
        auto* row_labels = labels.data() + static_cast<std::size_t>(y) * width;
        std::fill(best.begin(), best.end(), -std::numeric_limits<float>::infinity());
        for (std::int32_t label = 0; label < classes; ++label) {
            const float* source = scores + static_cast<std::size_t>(label) * plane;
            for (std::int32_t x = 0; x < width; ++x) {
                const auto left =
                    static_cast<std::size_t>(columns.low[static_cast<std::size_t>(x)]);
                const auto right =
                    static_cast<std::size_t>(columns.high[static_cast<std::size_t>(x)]);
                const float left_weight = columns.low_weight[static_cast<std::size_t>(x)];
                const float right_weight = columns.high_weight[static_cast<std::size_t>(x)];
                const float value = top_weight * (left_weight * source[top + left] +
                                                  right_weight * source[top + right]) +
                                    bottom_weight * (left_weight * source[bottom + left] +
                                                     right_weight * source[bottom + right]);
                if (!std::isfinite(value))
                    throw std::runtime_error("Mask2Former semantic scores are not finite");
                // Strictly greater keeps the first maximum, as torch.argmax does.
                if (value > best[static_cast<std::size_t>(x)]) {
                    best[static_cast<std::size_t>(x)] = value;
                    row_labels[x] = label;
                }
            }
        }
    }
    return labels;
}

} // namespace trtmc::mask2former
