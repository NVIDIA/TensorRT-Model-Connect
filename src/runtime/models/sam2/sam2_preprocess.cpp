/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::sam2 {

namespace {

constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowResizePrecisionBits - 1);
constexpr std::int64_t kPillowCoefficientScale = std::int64_t{1} << kPillowResizePrecisionBits;

std::size_t checkedImageElements(std::int32_t height, std::int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("SAM2 image dimensions must be positive");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    if (w > std::numeric_limits<std::size_t>::max() / h ||
        h * w > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::overflow_error("SAM2 image element count overflows");
    }
    return h * w * 3U;
}

double bicubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::uint8_t applySpan(const std::uint8_t* source, std::int32_t stride,
                       const PillowResizeAxisPlan& plan, const PillowResizeSpan& span) {
    std::int64_t sum = kPillowRounding;
    for (std::int32_t index = 0; index < span.weight_count; ++index) {
        sum += static_cast<std::int64_t>(source[static_cast<std::size_t>(span.first + index) *
                                                static_cast<std::size_t>(stride)]) *
               plan.weights[static_cast<std::size_t>(span.weight_offset + index)];
    }
    // Avoid implementation-defined right shift of a negative signed value.
    if (sum <= 0)
        return std::uint8_t{0};
    const auto rounded = sum >> kPillowResizePrecisionBits;
    if (rounded >= 255)
        return std::uint8_t{255};
    return static_cast<std::uint8_t>(rounded);
}

std::uint8_t floatToSourceByte(float value) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
        throw std::invalid_argument("SAM2 RGB input must contain finite values in [0, 1]");
    return static_cast<std::uint8_t>(std::lround(static_cast<double>(value) * 255.0));
}

} // namespace

PillowResizeAxisPlan makePillowBicubicAxisPlan(std::int32_t input_size, std::int32_t output_size) {
    if (input_size <= 0 || output_size <= 0)
        throw std::invalid_argument("SAM2 resize dimensions must be positive");
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    PillowResizeAxisPlan plan;
    plan.input_size = input_size;
    plan.output_size = output_size;
    plan.spans.resize(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("SAM2 Pillow resize produced empty filter support");

        auto& span = plan.spans[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weight_offset = static_cast<std::int32_t>(plan.weights.size());
        span.weight_count = end - first;
        const auto count = static_cast<std::size_t>(span.weight_count);
        std::vector<double> floating(count);
        double sum = 0.0;
        for (std::int32_t input_index = first; input_index < end; ++input_index) {
            const double distance =
                (static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale;
            const double weight = bicubic(distance);
            floating[static_cast<std::size_t>(input_index - first)] = weight;
            sum += weight;
        }
        if (!std::isfinite(sum) || sum == 0.0)
            throw std::runtime_error("SAM2 Pillow resize has invalid coefficient sum");

        plan.weights.reserve(plan.weights.size() + count);
        for (const double raw_weight : floating) {
            const double scaled = raw_weight / sum * static_cast<double>(kPillowCoefficientScale);
            if (!std::isfinite(scaled) ||
                scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("SAM2 Pillow resize coefficient overflowed");
            }
            plan.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

const Sam2Rgb8NormalizationTable& sam2Rgb8NormalizationTable() {
    static const Sam2Rgb8NormalizationTable table = [] {
        constexpr std::array<float, kSam2RgbChannels> kMean = {0.485F, 0.456F, 0.406F};
        constexpr std::array<float, kSam2RgbChannels> kStd = {0.229F, 0.224F, 0.225F};
        Sam2Rgb8NormalizationTable result{};
        for (std::size_t channel = 0; channel < kSam2RgbChannels; ++channel) {
            for (std::size_t value = 0; value < kSam2Rgb8ValueCount; ++value) {
                const float source_value = static_cast<float>(static_cast<double>(value) / 255.0);
                result[channel * kSam2Rgb8ValueCount + value] =
                    (source_value - kMean[channel]) / kStd[channel];
            }
        }
        return result;
    }();
    return table;
}

std::vector<std::uint8_t> resizePillowBicubicRgb(const std::uint8_t* input,
                                                 std::int32_t input_height,
                                                 std::int32_t input_width,
                                                 std::int32_t output_height,
                                                 std::int32_t output_width) {
    const auto input_elements = checkedImageElements(input_height, input_width);
    (void)checkedImageElements(output_height, output_width);
    if (input == nullptr)
        throw std::invalid_argument("SAM2 resize input must not be null");
    if (input_height == output_height && input_width == output_width)
        return {input, input + input_elements};

    const std::uint8_t* vertical_source = input;
    std::vector<std::uint8_t> horizontal;
    if (input_width != output_width) {
        const auto plan = makePillowBicubicAxisPlan(input_width, output_width);
        horizontal.resize(static_cast<std::size_t>(input_height) *
                          static_cast<std::size_t>(output_width) * 3U);
        for (std::int32_t y = 0; y < input_height; ++y) {
            for (std::int32_t x = 0; x < output_width; ++x) {
                const auto& span = plan.spans[static_cast<std::size_t>(x)];
                for (std::int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        static_cast<std::size_t>(y) * static_cast<std::size_t>(input_width) * 3U +
                        static_cast<std::size_t>(channel);
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(output_width) +
                         static_cast<std::size_t>(x)) *
                            3U +
                        static_cast<std::size_t>(channel);
                    horizontal[destination_offset] =
                        applySpan(input + source_offset, 3, plan, span);
                }
            }
        }
        vertical_source = horizontal.data();
    }
    if (input_height == output_height)
        return horizontal;

    const auto plan = makePillowBicubicAxisPlan(input_height, output_height);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_height) *
                                     static_cast<std::size_t>(output_width) * 3U);
    for (std::int32_t y = 0; y < output_height; ++y) {
        const auto& span = plan.spans[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < output_width; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset =
                    static_cast<std::size_t>(x) * 3U + static_cast<std::size_t>(channel);
                const auto destination_offset =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(output_width) +
                     static_cast<std::size_t>(x)) *
                        3U +
                    static_cast<std::size_t>(channel);
                output[destination_offset] =
                    applySpan(vertical_source + source_offset, output_width * 3, plan, span);
            }
        }
    }
    return output;
}

PreprocessedFrame preprocessFrame(const float* input, std::int32_t input_height,
                                  std::int32_t input_width) {
    const auto input_elements = checkedImageElements(input_height, input_width);
    if (input == nullptr)
        throw std::invalid_argument("SAM2 frame input must not be null");
    std::vector<std::uint8_t> source(input_elements);
    std::transform(input, input + input_elements, source.begin(), floatToSourceByte);

    return preprocessRgb8Frame(source.data(), input_height, input_width);
}

PreprocessedFrame preprocessRgb8Frame(const std::uint8_t* input, std::int32_t input_height,
                                      std::int32_t input_width) {
    (void)checkedImageElements(input_height, input_width);
    if (input == nullptr)
        throw std::invalid_argument("SAM2 RGB8 frame input must not be null");

    PreprocessedFrame output;
    output.resized_rgb_hwc = resizePillowBicubicRgb(input, input_height, input_width,
                                                    kPreprocessImageSize, kPreprocessImageSize);
    const auto& normalization = sam2Rgb8NormalizationTable();
    constexpr std::size_t kArea = static_cast<std::size_t>(kPreprocessImageSize) *
                                  static_cast<std::size_t>(kPreprocessImageSize);
    output.pixel_values.resize(kArea * 3U);
    for (std::int32_t y = 0; y < kPreprocessImageSize; ++y) {
        for (std::int32_t x = 0; x < kPreprocessImageSize; ++x) {
            const auto pixel =
                (static_cast<std::size_t>(y) * kPreprocessImageSize + static_cast<std::size_t>(x));
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                const auto value = output.resized_rgb_hwc[pixel * 3U + channel];
                output.pixel_values[channel * kArea + pixel] =
                    normalization[channel * kSam2Rgb8ValueCount + value];
            }
        }
    }
    return output;
}

} // namespace trtmc::sam2
