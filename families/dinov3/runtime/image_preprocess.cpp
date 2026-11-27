/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov3/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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

// The same arithmetic as the reference torchvision-style resize, per output
// element in the same order (an accumulator from 0.0F, each tap added in span
// order), restructured for the CPU: the horizontal pass keeps one accumulator
// per channel of a pixel (three independent chains), the vertical pass sweeps
// a whole output row per tap (contiguous, vectorizable) into a one-row
// accumulator and normalizes it into the NCHW planes as it goes; the full-size
// horizontal buffer is never zero-filled first.
void resize_rows(const float* input, int32_t input_h, int32_t input_w, int32_t output_w,
                 const std::vector<BilinearSpan>& spans, float* horizontal) {
    const auto row_in = static_cast<std::size_t>(input_w) * 3U;
    const auto row_out = static_cast<std::size_t>(output_w) * 3U;
    for (int32_t y = 0; y < input_h; ++y) {
        const float* row = input + static_cast<std::size_t>(y) * row_in;
        float* out = horizontal + static_cast<std::size_t>(y) * row_out;
        for (int32_t x = 0; x < output_w; ++x) {
            const auto& span = spans[static_cast<std::size_t>(x)];
            const float* source = row + static_cast<std::size_t>(span.first) * 3U;
            float r = 0.0F;
            float g = 0.0F;
            float b = 0.0F;
            for (std::size_t i = 0; i < span.weights.size(); ++i) {
                const float weight = span.weights[i];
                r += source[i * 3U] * weight;
                g += source[i * 3U + 1U] * weight;
                b += source[i * 3U + 2U] * weight;
            }
            out[static_cast<std::size_t>(x) * 3U] = r;
            out[static_cast<std::size_t>(x) * 3U + 1U] = g;
            out[static_cast<std::size_t>(x) * 3U + 2U] = b;
        }
    }
}

void normalize_row(const float* row, int32_t y, int32_t output_w, std::size_t output_plane,
                   const Dinov3PreprocessConfig& config, float* pixel_values) {
    const float mean[3] = {config.image_mean[0], config.image_mean[1], config.image_mean[2]};
    const float std_dev[3] = {config.image_std[0], config.image_std[1], config.image_std[2]};
    const auto base = static_cast<std::size_t>(y) * output_w;
    for (int32_t x = 0; x < output_w; ++x) {
        for (int32_t c = 0; c < 3; ++c) {
            pixel_values[static_cast<std::size_t>(c) * output_plane + base + x] =
                (row[static_cast<std::size_t>(x) * 3U + c] - mean[c]) / std_dev[c];
        }
    }
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
    const auto row_out = static_cast<std::size_t>(output_w) * 3U;
    const auto output_plane = static_cast<std::size_t>(output_h) * output_w;

    std::unique_ptr<float[]> horizontal;
    const float* vertical_input = image_pixels;
    if (image_width != output_w) {
        horizontal.reset(new float[static_cast<std::size_t>(image_height) * row_out]);
        resize_rows(image_pixels, image_height, image_width, output_w,
                    make_bilinear_spans(image_width, output_w), horizontal.get());
        vertical_input = horizontal.get();
    }

    std::vector<float> pixel_values(3U * output_plane);
    if (image_height == output_h) {
        for (int32_t y = 0; y < output_h; ++y)
            normalize_row(vertical_input + static_cast<std::size_t>(y) * row_out, y, output_w,
                          output_plane, config, pixel_values.data());
        return pixel_values;
    }
    const auto spans = make_bilinear_spans(image_height, output_h);
    std::unique_ptr<float[]> row(new float[row_out]);
    for (int32_t y = 0; y < output_h; ++y) {
        const auto& span = spans[static_cast<std::size_t>(y)];
        std::fill(row.get(), row.get() + row_out, 0.0F);
        for (std::size_t i = 0; i < span.weights.size(); ++i) {
            const float weight = span.weights[i];
            const float* source =
                vertical_input +
                static_cast<std::size_t>(span.first + static_cast<int32_t>(i)) * row_out;
            for (std::size_t k = 0; k < row_out; ++k)
                row[k] += source[k] * weight;
        }
        normalize_row(row.get(), y, output_w, output_plane, config, pixel_values.data());
    }
    return pixel_values;
}

} // namespace trtmc
