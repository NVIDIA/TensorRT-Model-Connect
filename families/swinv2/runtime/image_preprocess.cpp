/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/swinv2/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::swinv2 {

namespace {

// Pillow's bicubic resampler, reproduced coefficient for coefficient. The
// reference processor resizes 8-bit pixels with Pillow, and a float resize or a
// differently normalised kernel lands a few least-significant bits away on
// every pixel.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);
constexpr double kBicubicSupport = 2.0;

struct PillowSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double pillow_bicubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::int32_t fixed_point(double weight, double total) {
    const double scaled = weight / total * static_cast<double>(kPillowScale);
    if (!std::isfinite(scaled) ||
        scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("swinv2 Pillow resize coefficient overflowed");
    }
    return static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
}

std::vector<PillowSpan> make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = kBicubicSupport * filter_scale;
    // Pillow multiplies by the reciprocal rather than dividing; the two can
    // differ in the last bit of a coefficient.
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<PillowSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("swinv2 Pillow resize produced empty support");

        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_bicubic((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("swinv2 Pillow resize has invalid coefficients");
        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weights.reserve(floating.size());
        for (double weight : floating)
            span.weights.push_back(fixed_point(weight, total));
    }
    return plan;
}

std::uint8_t apply_pillow_span(const std::uint8_t* source, std::int32_t stride,
                               const PillowSpan& span) {
    std::int64_t sum = kPillowRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        sum += static_cast<std::int64_t>(
                   source[(span.first + static_cast<std::int32_t>(index)) * stride]) *
               span.weights[index];
    }
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPillowPrecisionBits, 255));
}

// One pass along the rows: [height, input_w, 3] -> [height, output_w, 3], rounded to 8 bits.
std::vector<std::uint8_t> horizontal_pass(const std::vector<std::uint8_t>& image,
                                          std::int32_t height, std::int32_t input_w,
                                          std::int32_t output_w) {
    const auto plan = make_pillow_plan(input_w, output_w);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(height) * output_w * 3U);
    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < output_w; ++x) {
            const auto& span = plan[static_cast<std::size_t>(x)];
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset = static_cast<std::size_t>(y) * input_w * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[target] = apply_pillow_span(image.data() + source_offset, 3, span);
            }
        }
    }
    return output;
}

// One pass along the columns: [input_h, width, 3] -> [output_h, width, 3], rounded to 8 bits.
std::vector<std::uint8_t> vertical_pass(const std::vector<std::uint8_t>& image,
                                        std::int32_t input_h, std::int32_t width,
                                        std::int32_t output_h) {
    const auto plan = make_pillow_plan(input_h, output_h);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_h) * width * 3U);
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto& span = plan[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < width; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset = static_cast<std::size_t>(x) * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * width + x) * 3U + channel;
                output[target] = apply_pillow_span(image.data() + source_offset, width * 3, span);
            }
        }
    }
    return output;
}

// Two separable passes, each rounding to 8 bits, in the order Pillow 13 picks:
// the vertical pass runs first when the height shrinks by more than twice as
// many rows as the width loses columns, the horizontal pass first otherwise
// (Pillow 12 and earlier always ran the horizontal pass first). An axis whose
// size does not change is not resampled.
std::vector<std::uint8_t> resize_pillow(const float* source, std::int32_t input_h,
                                        std::int32_t input_w, std::int32_t output_h,
                                        std::int32_t output_w) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(input_h) * input_w * 3U);
    std::transform(source, source + image.size(), image.begin(), [](float value) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    });

    const bool horizontal_first =
        !((input_h - output_h) > 0 && (input_h - output_h) > (input_w - output_w) * 2);
    if (horizontal_first) {
        if (input_w != output_w)
            image = horizontal_pass(image, input_h, input_w, output_w);
        if (input_h != output_h)
            image = vertical_pass(image, input_h, output_w, output_h);
    } else {
        if (input_h != output_h)
            image = vertical_pass(image, input_h, input_w, output_h);
        if (input_w != output_w)
            image = horizontal_pass(image, output_h, input_w, output_w);
    }
    return image;
}

void validate(const PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("swinv2 input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("swinv2 image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("swinv2 image std must be non-zero");
    }
}

} // namespace

std::vector<float> preprocess_image(const float* image_pixels, std::int32_t image_height,
                                    std::int32_t image_width, const PreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("swinv2 source image must be non-empty");
    validate(config);

    // Straight to the target in both axes: every pixel of the original is
    // represented, squashed. Preserving the aspect ratio would feed the model a
    // different image than the reference saw.
    const auto resized = resize_pillow(image_pixels, image_height, image_width,
                                       config.input_image_h, config.input_image_w);

    // The reference rescales in double and rounds to float32 once, then
    // normalizes in float32; the same two steps here give the same pixels.
    constexpr double kRescale = 1.0 / 255.0;
    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::size_t index = 0; index < plane; ++index) {
        for (std::size_t channel = 0; channel < 3U; ++channel) {
            const auto value =
                static_cast<float>(static_cast<double>(resized[index * 3U + channel]) * kRescale);
            pixel_values[channel * plane + index] =
                (value - config.image_mean[channel]) / config.image_std[channel];
        }
    }
    return pixel_values;
}

} // namespace trtmc::swinv2
