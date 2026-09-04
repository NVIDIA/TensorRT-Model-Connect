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

double cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

void validate_preprocess_config(const Dinov3PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("DINOv3 input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("DINOv3 image mean/std must contain three channels");
    if (config.interpolation != "bilinear" && config.interpolation != "bicubic")
        throw std::invalid_argument("DINOv3 interpolation must be bilinear or bicubic");
    if (config.crop_pct <= 0.0F || config.crop_pct > 1.0F)
        throw std::invalid_argument("DINOv3 crop_pct must be in (0, 1]");
    for (float value : config.image_std) {
        if (!std::isfinite(value) || value == 0.0F)
            throw std::invalid_argument("DINOv3 image std must be finite and non-zero");
    }
}

std::vector<BilinearSpan> make_bicubic_spans(int32_t input_size, int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<BilinearSpan> spans(static_cast<std::size_t>(output_size));
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const int32_t first = std::max(0, static_cast<int32_t>(center - support + 0.5));
        const int32_t end = std::min(input_size, static_cast<int32_t>(center + support + 0.5));
        auto& span = spans[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weights.resize(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (int32_t input_index = first; input_index < end; ++input_index) {
            const double weight =
                cubic((static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale);
            span.weights[static_cast<std::size_t>(input_index - first)] =
                static_cast<float>(weight);
            total += weight;
        }
        if (total == 0.0)
            throw std::runtime_error("DINOv3 bicubic resize has empty support");
        for (float& weight : span.weights)
            weight = static_cast<float>(static_cast<double>(weight) / total);
    }
    return spans;
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

std::vector<float> resize_bicubic_rgb(const float* input, int32_t input_h, int32_t input_w,
                                      int32_t output_h, int32_t output_w) {
    std::vector<float> horizontal;
    const float* vertical_input = input;
    if (input_w != output_w) {
        const auto spans = make_bicubic_spans(input_w, output_w);
        horizontal.resize(static_cast<std::size_t>(input_h) * output_w * 3U);
        for (int32_t y = 0; y < input_h; ++y) {
            for (int32_t x = 0; x < output_w; ++x) {
                const auto& span = spans[static_cast<std::size_t>(x)];
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source = static_cast<std::size_t>(y) * input_w * 3U + channel;
                    const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                    horizontal[target] = apply_bilinear_span(input + source, 3, span);
                }
            }
        }
        vertical_input = horizontal.data();
    }
    if (input_h == output_h)
        return input_w == output_w
                   ? std::vector<float>(input,
                                        input + static_cast<std::size_t>(input_h) * input_w * 3U)
                   : horizontal;

    const auto spans = make_bicubic_spans(input_h, output_h);
    std::vector<float> output(static_cast<std::size_t>(output_h) * output_w * 3U);
    for (int32_t y = 0; y < output_h; ++y) {
        const auto& span = spans[static_cast<std::size_t>(y)];
        for (int32_t x = 0; x < output_w; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(x) * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[target] = apply_bilinear_span(vertical_input + source, output_w * 3, span);
            }
        }
    }
    return output;
}

int32_t center_crop_offset(int32_t resized, int32_t target) {
    const int32_t difference = resized - target;
    const int32_t half = difference / 2;
    return (difference % 2 != 0 && half % 2 != 0) ? half + 1 : half;
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
    int32_t resized_h = output_h;
    int32_t resized_w = output_w;
    if (config.do_center_crop) {
        const int32_t resized_short =
            static_cast<int32_t>(std::floor(static_cast<float>(output_h) / config.crop_pct));
        if (image_height <= image_width) {
            resized_h = resized_short;
            resized_w = static_cast<int32_t>(static_cast<int64_t>(resized_short) * image_width /
                                             image_height);
        } else {
            resized_h = static_cast<int32_t>(static_cast<int64_t>(resized_short) * image_height /
                                             image_width);
            resized_w = resized_short;
        }
    }
    const auto resized =
        config.interpolation == "bicubic"
            ? resize_bicubic_rgb(image_pixels, image_height, image_width, resized_h, resized_w)
            : resize_torchvision_bilinear_rgb(image_pixels, image_height, image_width, resized_h,
                                              resized_w);
    const int32_t crop_y = config.do_center_crop ? center_crop_offset(resized_h, output_h) : 0;
    const int32_t crop_x = config.do_center_crop ? center_crop_offset(resized_w, output_w) : 0;

    const auto output_plane = static_cast<std::size_t>(output_h) * output_w;
    std::vector<float> pixel_values(3U * output_plane);
    for (int32_t y = 0; y < output_h; ++y) {
        for (int32_t x = 0; x < output_w; ++x) {
            const auto src_idx =
                static_cast<std::size_t>(((crop_y + y) * resized_w + crop_x + x) * 3);
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
