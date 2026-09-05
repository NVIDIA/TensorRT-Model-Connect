/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov3/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace {

struct BilinearSpan {
    int32_t first{0};
    std::vector<float> weights;
};

void validate_preprocess_config(const Dinov3PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("DINOv3 input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("DINOv3 image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (!std::isfinite(value) || value == 0.0F)
            throw std::invalid_argument("DINOv3 image std must be finite and non-zero");
    }
}

std::vector<BilinearSpan> make_bilinear_spans(int32_t input_size, int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<BilinearSpan> spans(static_cast<std::size_t>(output_size));

    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const int32_t first = std::max(0, static_cast<int32_t>(center - support + 0.5));
        const int32_t end = std::min(input_size, static_cast<int32_t>(center + support + 0.5));
        auto& span = spans[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weights.resize(static_cast<std::size_t>(end - first));

        double weight_sum = 0.0;
        for (int32_t input_index = first; input_index < end; ++input_index) {
            const double distance =
                std::abs((static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale);
            const double weight = distance < 1.0 ? 1.0 - distance : 0.0;
            span.weights[static_cast<std::size_t>(input_index - first)] =
                static_cast<float>(weight);
            weight_sum += weight;
        }
        for (float& weight : span.weights)
            weight = static_cast<float>(static_cast<double>(weight) / weight_sum);
    }
    return spans;
}

float apply_bilinear_span(const float* input, int32_t stride, const BilinearSpan& span) {
    float result = 0.0F;
    for (std::size_t i = 0; i < span.weights.size(); ++i) {
        result += input[(span.first + static_cast<int32_t>(i)) * stride] * span.weights[i];
    }
    return result;
}

std::vector<float> resize_torchvision_bilinear_rgb(const float* input, int32_t input_h,
                                                   int32_t input_w, int32_t output_h,
                                                   int32_t output_w) {
    std::vector<float> horizontal;
    const float* vertical_input = input;
    if (input_w != output_w) {
        const auto spans = make_bilinear_spans(input_w, output_w);
        horizontal.resize(static_cast<std::size_t>(input_h) * output_w * 3U);
        for (int32_t y = 0; y < input_h; ++y) {
            for (int32_t x = 0; x < output_w; ++x) {
                const auto& span = spans[static_cast<std::size_t>(x)];
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto input_offset = static_cast<std::size_t>(y) * input_w * 3U +
                                              static_cast<std::size_t>(channel);
                    const auto output_offset =
                        (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                    horizontal[output_offset] = apply_bilinear_span(input + input_offset, 3, span);
                }
            }
        }
        vertical_input = horizontal.data();
    }

    if (input_h == output_h) {
        if (input_w == output_w) {
            return std::vector<float>(input,
                                      input + static_cast<std::size_t>(input_h) * input_w * 3U);
        }
        return horizontal;
    }

    const auto spans = make_bilinear_spans(input_h, output_h);
    std::vector<float> output(static_cast<std::size_t>(output_h) * output_w * 3U);
    for (int32_t y = 0; y < output_h; ++y) {
        const auto& span = spans[static_cast<std::size_t>(y)];
        for (int32_t x = 0; x < output_w; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto input_offset = static_cast<std::size_t>(x) * 3U + channel;
                const auto output_offset =
                    (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[output_offset] =
                    apply_bilinear_span(vertical_input + input_offset, output_w * 3, span);
            }
        }
    }
    return output;
}

} // namespace

std::vector<float> preprocess_dinov3_image(const float* image_pixels, int32_t image_height,
                                           int32_t image_width,
                                           const Dinov3PreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("DINOv3 source image must be non-empty");
    validate_preprocess_config(config);

    const int32_t output_h = config.input_image_h;
    const int32_t output_w = config.input_image_w;
    const auto resized = resize_torchvision_bilinear_rgb(image_pixels, image_height, image_width,
                                                         output_h, output_w);

    const auto output_plane = static_cast<std::size_t>(output_h) * output_w;
    std::vector<float> pixel_values(3U * output_plane);
    for (int32_t y = 0; y < output_h; ++y) {
        for (int32_t x = 0; x < output_w; ++x) {
            const auto src_idx = static_cast<std::size_t>((y * output_w + x) * 3);
            const auto pixel_idx = static_cast<std::size_t>(y) * output_w + x;
            for (int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                pixel_values[channel * output_plane + pixel_idx] =
                    (resized[src_idx + channel] - config.image_mean[channel]) /
                    config.image_std[channel];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
