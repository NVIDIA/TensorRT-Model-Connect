/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/idefics3/runtime/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

// Pillow LANCZOS resampling with its signed 22-bit fixed-point coefficients.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);

struct PillowSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double pillow_lanczos(double value) {
    value = std::abs(value);
    if (value == 0) return 1;
    if (value >= 3) return 0;
    constexpr double pi = 3.14159265358979323846;
    return std::sin(pi * value) / (pi * value) * std::sin(pi * value / 3) / (pi * value / 3);
}

std::vector<PillowSpan> make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 3.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<PillowSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("idefics3 Pillow resize produced empty support");

        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_lanczos((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("idefics3 Pillow resize has invalid coefficients");
        span.weights.reserve(floating.size());
        for (double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kPillowScale);
            if (!std::isfinite(scaled) ||
                scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("idefics3 Pillow resize coefficient overflowed");
            }
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
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

std::vector<float> resize_rgb(const float* source, std::int32_t input_h,
                                         std::int32_t input_w, std::int32_t output_h,
                                         std::int32_t output_w) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(input_h) * input_w * 3U);
    std::transform(source, source + input.size(), input.begin(), [](float value) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    });

    const auto horizontal_plan = make_pillow_plan(input_w, output_w);
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(input_h) * output_w * 3U);
    for (std::int32_t y = 0; y < input_h; ++y) {
        for (std::int32_t x = 0; x < output_w; ++x) {
            const auto& span = horizontal_plan[static_cast<std::size_t>(x)];
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset = static_cast<std::size_t>(y) * input_w * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                horizontal[target] = apply_pillow_span(input.data() + source_offset, 3, span);
            }
        }
    }

    const auto vertical_plan = make_pillow_plan(input_h, output_h);
    std::vector<float> output(static_cast<std::size_t>(output_h) * output_w * 3U);
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto& span = vertical_plan[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < output_w; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset = static_cast<std::size_t>(x) * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[target] = static_cast<float>(apply_pillow_span(
                                     horizontal.data() + source_offset, output_w * 3, span)) /
                                 255.0F;
            }
        }
    }
    return output;
}


} // namespace

Idefics3Image preprocess_idefics3_image(const float* rgb, int32_t height, int32_t width,
                                      int32_t longest_edge, int32_t view_size) {
    if (!rgb || height <= 0 || width <= 0 || longest_edge <= 0 || view_size <= 0)
        throw std::invalid_argument("Invalid Idefics3 image dimensions");
    int32_t resized_h, resized_w;
    const double aspect = double(width) / height;
    if (width >= height) {
        resized_w = longest_edge;
        resized_h = static_cast<int32_t>(resized_w / aspect);
        if (resized_h % 2) ++resized_h;
    } else {
        resized_h = longest_edge;
        resized_w = static_cast<int32_t>(resized_h * aspect);
        if (resized_w % 2) ++resized_w;
    }
    resized_h = std::max(resized_h, 1); resized_w = std::max(resized_w, 1);
    const auto first = resize_rgb(rgb, height, width, resized_h, resized_w);
    const double resized_aspect = double(resized_w) / resized_h;
    int32_t padded_h, padded_w;
    if (resized_w >= resized_h) {
        padded_w = (resized_w + view_size - 1) / view_size * view_size;
        padded_h = static_cast<int32_t>(padded_w / resized_aspect);
        padded_h = (padded_h + view_size - 1) / view_size * view_size;
    } else {
        padded_h = (resized_h + view_size - 1) / view_size * view_size;
        padded_w = static_cast<int32_t>(padded_h * resized_aspect);
        padded_w = (padded_w + view_size - 1) / view_size * view_size;
    }
    const auto resized = resize_rgb(first.data(), resized_h, resized_w, padded_h, padded_w);
    Idefics3Image result;
    const auto plane = static_cast<std::size_t>(view_size) * view_size;
    auto append = [&](const float* source, int32_t stride, int32_t top, int32_t left) {
        std::vector<float> pixels(3 * plane);
        for (int32_t y = 0; y < view_size; ++y)
            for (int32_t x = 0; x < view_size; ++x)
                for (int32_t c = 0; c < 3; ++c)
                    pixels[c * plane + y * view_size + x] =
                        (source[((top + y) * stride + left + x) * 3 + c] - 0.5F) / 0.5F;
        result.views.push_back(std::move(pixels));
    };
    if (padded_h > view_size || padded_w > view_size) {
        result.rows = padded_h / view_size; result.columns = padded_w / view_size;
        for (int32_t row = 0; row < result.rows; ++row)
            for (int32_t column = 0; column < result.columns; ++column)
                append(resized.data(), padded_w, row * view_size, column * view_size);
        const auto global = resize_rgb(resized.data(), padded_h, padded_w, view_size, view_size);
        append(global.data(), view_size, 0, 0);
    } else append(resized.data(), padded_w, 0, 0);
    return result;
}

std::string idefics3_image_prompt(int32_t rows, int32_t columns, int32_t sequence_length) {
    std::string images;
    for (int32_t i = 0; i < sequence_length; ++i) images += "<image>";
    std::string result;
    for (int32_t row = 1; row <= rows; ++row) {
        for (int32_t column = 1; column <= columns; ++column)
            result += "<fake_token_around_image><row_" + std::to_string(row) + "_col_" +
                std::to_string(column) + ">" + images;
        result += '\n';
    }
    if (rows) result += '\n';
    return result + "<fake_token_around_image><global-img>" + images + "<fake_token_around_image>";
}

} // namespace trtmc
