/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/segformer/segformer_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

constexpr int32_t kPillowPrecisionBits = 22;
constexpr int32_t kPillowRounding = 1 << (kPillowPrecisionBits - 1);
constexpr int32_t kPillowCoefficientScale = 1 << kPillowPrecisionBits;

struct BilinearSpan {
    int32_t first{0};
    std::vector<int32_t> weights;
};

void validate_segformer_preprocess_config(const SegformerPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("SegFormer input dimensions must be positive");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("SegFormer image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("SegFormer image std must be non-zero");
    }
}

std::vector<BilinearSpan> make_pillow_bilinear_spans(int32_t input_size, int32_t output_size) {
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

        std::vector<double> floating_weights(span.weights.size());
        double weight_sum = 0.0;
        for (int32_t input_index = first; input_index < end; ++input_index) {
            const double distance =
                std::abs((static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale);
            const double weight = distance < 1.0 ? 1.0 - distance : 0.0;
            floating_weights[static_cast<std::size_t>(input_index - first)] = weight;
            weight_sum += weight;
        }
        for (std::size_t i = 0; i < span.weights.size(); ++i) {
            span.weights[i] = static_cast<int32_t>(0.5 + floating_weights[i] / weight_sum *
                                                             kPillowCoefficientScale);
        }
    }
    return spans;
}

uint8_t apply_pillow_bilinear_span(const uint8_t* input, int32_t stride, const BilinearSpan& span) {
    int64_t sum = kPillowRounding;
    for (std::size_t i = 0; i < span.weights.size(); ++i) {
        sum += static_cast<int64_t>(input[(span.first + static_cast<int32_t>(i)) * stride]) *
               span.weights[i];
    }
    return static_cast<uint8_t>(std::clamp<int64_t>(sum >> kPillowPrecisionBits, 0, 255));
}

std::vector<uint8_t> resize_pillow_bilinear_rgb(const std::vector<uint8_t>& input, int32_t input_h,
                                                int32_t input_w, int32_t output_h,
                                                int32_t output_w) {
    std::vector<uint8_t> horizontal;
    const std::vector<uint8_t>* vertical_input = &input;
    if (input_w != output_w) {
        const auto spans = make_pillow_bilinear_spans(input_w, output_w);
        horizontal.resize(static_cast<std::size_t>(input_h) * output_w * 3U);
        for (int32_t y = 0; y < input_h; ++y) {
            for (int32_t x = 0; x < output_w; ++x) {
                const auto& span = spans[static_cast<std::size_t>(x)];
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto input_offset = static_cast<std::size_t>(y) * input_w * 3U +
                                              static_cast<std::size_t>(channel);
                    const auto output_offset =
                        (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                    horizontal[output_offset] =
                        apply_pillow_bilinear_span(input.data() + input_offset, 3, span);
                }
            }
        }
        vertical_input = &horizontal;
    }

    if (input_h == output_h) {
        return *vertical_input;
    }
    const auto spans = make_pillow_bilinear_spans(input_h, output_h);
    std::vector<uint8_t> output(static_cast<std::size_t>(output_h) * output_w * 3U);
    for (int32_t y = 0; y < output_h; ++y) {
        const auto& span = spans[static_cast<std::size_t>(y)];
        for (int32_t x = 0; x < output_w; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto input_offset = (static_cast<std::size_t>(x) * 3U) + channel;
                const auto output_offset =
                    (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[output_offset] = apply_pillow_bilinear_span(
                    vertical_input->data() + input_offset, output_w * 3, span);
            }
        }
    }
    return output;
}

} // namespace

std::vector<float> preprocess_segformer_image(const float* image_pixels, int32_t image_height,
                                              int32_t image_width,
                                              const SegformerPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("SegFormer source image must be non-empty");
    }
    validate_segformer_preprocess_config(config);

    const int32_t input_h = config.input_image_h;
    const int32_t input_w = config.input_image_w;
    const auto source_size = static_cast<std::size_t>(image_height) * image_width * 3U;
    std::vector<uint8_t> source_u8(source_size);
    for (std::size_t i = 0; i < source_size; ++i) {
        const float scaled = std::clamp(image_pixels[i], 0.0F, 1.0F) * 255.0F;
        source_u8[i] = static_cast<uint8_t>(std::lround(scaled));
    }

    // Hugging Face's SegformerImageProcessor resizes PIL RGB images before
    // converting them to float tensors. Match Pillow's separable bilinear
    // filter support, half-pixel centers, fixed-point coefficients, and uint8
    // rounding at each pass.
    const auto resized =
        resize_pillow_bilinear_rgb(source_u8, image_height, image_width, input_h, input_w);

    std::vector<float> pixel_values(static_cast<std::size_t>(3) * input_h * input_w);
    for (int32_t y = 0; y < input_h; ++y) {
        for (int32_t x = 0; x < input_w; ++x) {
            const auto src_idx = static_cast<std::size_t>((y * input_w + x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                const float rescaled = static_cast<float>(resized[src_idx + channel]) / 255.0F;
                const float value =
                    (rescaled - config.image_mean[channel]) / config.image_std[channel];
                pixel_values[channel * static_cast<std::size_t>(input_h) * input_w +
                             static_cast<std::size_t>(y) * input_w + x] = value;
            }
        }
    }
    return pixel_values;
}

std::vector<float> preprocess_segformer_image(const runtime::adapters::io::DecodedImage& image,
                                              const SegformerPreprocessConfig& config) {
    if (image.empty() || image.channels < 3) {
        throw std::invalid_argument("SegFormer decoded image must contain RGB pixels");
    }

    std::vector<float> pixels(static_cast<std::size_t>(image.height) * image.width * 3U);
    for (int32_t y = 0; y < image.height; ++y) {
        for (int32_t x = 0; x < image.width; ++x) {
            const auto src_idx = static_cast<std::size_t>((y * image.width + x) * image.channels);
            const auto dst_idx = static_cast<std::size_t>((y * image.width + x) * 3);
            for (int32_t c = 0; c < 3; ++c) {
                pixels[dst_idx + static_cast<std::size_t>(c)] =
                    static_cast<float>(image.pixels[src_idx + static_cast<std::size_t>(c)]) /
                    255.0F;
            }
        }
    }
    return preprocess_segformer_image(pixels.data(), image.height, image.width, config);
}

} // namespace trtmc
