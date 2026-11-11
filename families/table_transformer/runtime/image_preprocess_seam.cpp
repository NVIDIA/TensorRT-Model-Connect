/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/table_transformer/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

void validate_table_transformer_preprocess_config(const TableTransformerPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("table_transformer input dimensions must be positive");
    }
    if (config.shortest_edge <= 0 || config.longest_edge <= 0) {
        throw std::invalid_argument("table_transformer resize edges must be positive");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("table_transformer image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F) {
            throw std::invalid_argument("table_transformer image std must be non-zero");
        }
    }
}

int32_t hf_round(double value) {
    // Python's round() chooses the even integer at a half tie.
    const auto lower = static_cast<int32_t>(std::floor(value));
    const double fraction = value - lower;
    return lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0));
}

struct TableTransformerResizePlan {
    double raw_size{0.0};
    bool has_raw_size{false};
    int32_t short_edge{0};
};

TableTransformerResizePlan
compute_table_transformer_resize_plan(double height, double width,
                                      const TableTransformerPreprocessConfig& config) {
    const double size = static_cast<double>(config.shortest_edge);
    const double max_size = static_cast<double>(config.longest_edge);
    const double min_original = std::min(height, width);
    const double max_original = std::max(height, width);
    TableTransformerResizePlan plan;
    plan.short_edge = config.shortest_edge;
    if (max_original / min_original * size > max_size) {
        plan.raw_size = max_size * min_original / max_original;
        plan.has_raw_size = true;
        plan.short_edge = hf_round(plan.raw_size);
    }
    return plan;
}

int32_t resize_other_axis(const TableTransformerResizePlan& plan, double size, double primary,
                          double secondary) {
    if (plan.has_raw_size) {
        return static_cast<int32_t>(plan.raw_size * secondary / primary);
    }
    return static_cast<int32_t>(size * secondary / primary);
}

// The PIL bilinear sampling contract is documented by its reference source:
// https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c
// This family implementation owns its coefficient storage and RGB traversal.
constexpr int kFilterBits = 22;
struct FilterRow {
    int start;
    std::vector<int32_t> coefficients;
};

std::vector<FilterRow> bilinear_weights(int source, int destination) {
    const double scale = static_cast<double>(source) / destination;
    const double support = std::max(1.0, scale);
    std::vector<FilterRow> rows;
    rows.reserve(destination);
    for (int index = 0; index < destination; ++index) {
        const double center = (index + 0.5) * scale;
        const int first = std::max(0, static_cast<int>(center - support + 0.5));
        const int end = std::min(source, static_cast<int>(center + support + 0.5));
        std::vector<double> values;
        double total = 0.0;
        for (int position = first; position < end; ++position) {
            const double weight =
                std::max(0.0, 1.0 - std::abs((position - center + 0.5) / support));
            values.push_back(weight);
            total += weight;
        }
        FilterRow row{first, {}};
        for (double value : values)
            row.coefficients.push_back(
                static_cast<int32_t>(value / total * (1 << kFilterBits) + 0.5));
        rows.push_back(std::move(row));
    }
    return rows;
}

std::vector<uint8_t> resize_rgb_bytes(const float* pixels, int height, int width, int out_height,
                                      int out_width) {
    std::vector<uint8_t> source(static_cast<std::size_t>(height) * width * 3);
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (!std::isfinite(pixels[i]) || pixels[i] < 0.0F || pixels[i] > 1.0F)
            throw std::invalid_argument("table_transformer requires RGB values in [0,1]");
        source[i] = static_cast<uint8_t>(std::lround(pixels[i] * 255.0F));
    }
    const auto horizontal = bilinear_weights(width, out_width);
    const auto vertical = bilinear_weights(height, out_height);
    std::vector<uint8_t> intermediate(static_cast<std::size_t>(height) * out_width * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            const auto& row = horizontal[x];
            for (int channel = 0; channel < 3; ++channel) {
                int64_t sum = 1 << (kFilterBits - 1);
                for (std::size_t k = 0; k < row.coefficients.size(); ++k)
                    sum += source[(static_cast<std::size_t>(y) * width + row.start + k) * 3 +
                                  channel] *
                           static_cast<int64_t>(row.coefficients[k]);
                intermediate[(static_cast<std::size_t>(y) * out_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> kFilterBits, 0, 255));
            }
        }
    }
    std::vector<uint8_t> result(static_cast<std::size_t>(out_height) * out_width * 3);
    for (int y = 0; y < out_height; ++y) {
        const auto& row = vertical[y];
        for (int x = 0; x < out_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                int64_t sum = 1 << (kFilterBits - 1);
                for (std::size_t k = 0; k < row.coefficients.size(); ++k)
                    sum += intermediate[((row.start + k) * out_width + x) * 3 + channel] *
                           static_cast<int64_t>(row.coefficients[k]);
                result[(static_cast<std::size_t>(y) * out_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<int64_t>(sum >> kFilterBits, 0, 255));
            }
        }
    }
    return result;
}

} // namespace

TableTransformerResizeShape
compute_table_transformer_resize_shape(int32_t image_height, int32_t image_width,
                                       const TableTransformerPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("table_transformer source dimensions must be positive");
    }
    validate_table_transformer_preprocess_config(config);

    const double size = static_cast<double>(config.shortest_edge);
    const double height = static_cast<double>(image_height);
    const double width = static_cast<double>(image_width);
    const TableTransformerResizePlan plan =
        compute_table_transformer_resize_plan(height, width, config);

    if (std::min(height, width) == plan.short_edge) {
        return {image_height, image_width};
    }
    if (width < height) {
        return {resize_other_axis(plan, size, width, height), plan.short_edge};
    }
    return {plan.short_edge, resize_other_axis(plan, size, height, width)};
}

std::vector<float>
preprocess_table_transformer_image(const float* image_pixels, int32_t image_height,
                                   int32_t image_width,
                                   const TableTransformerPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("table_transformer source image must be non-empty");
    }
    validate_table_transformer_preprocess_config(config);

    const auto resize_shape =
        compute_table_transformer_resize_shape(image_height, image_width, config);
    const int32_t resized_h = resize_shape.height;
    const int32_t resized_w = resize_shape.width;
    if (resized_h > config.input_image_h || resized_w > config.input_image_w) {
        throw std::invalid_argument(
            "table_transformer input resize dimensions exceed the engine input dimensions");
    }
    if (!config.dynamic_image_shapes &&
        (resized_h != config.input_image_h || resized_w != config.input_image_w)) {
        throw std::invalid_argument("table_transformer input resize dimensions require padding, "
                                    "but the engine has no pixel_mask input");
    }

    const int32_t out_h = resized_h;
    const int32_t out_w = resized_w;

    // Match the official PIL RGB processor: antialiased separable bilinear
    // interpolation with 22-bit coefficients and byte rounding after each pass.
    // Floating-point STB resizing omits those byte boundaries and changes boxes.
    const auto resized =
        resize_rgb_bytes(image_pixels, image_height, image_width, resized_h, resized_w);

    const auto output_plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> pixel_values(3U * output_plane);

    for (int32_t y = 0; y < resized_h; ++y) {
        for (int32_t x = 0; x < resized_w; ++x) {
            const auto src_idx = static_cast<std::size_t>(((y * resized_w) + x) * 3);
            const float r =
                (resized[src_idx] / 255.0F - config.image_mean[0]) / config.image_std[0];
            const float g =
                (resized[src_idx + 1] / 255.0F - config.image_mean[1]) / config.image_std[1];
            const float b =
                (resized[src_idx + 2] / 255.0F - config.image_mean[2]) / config.image_std[2];
            pixel_values[static_cast<std::size_t>(y) * out_w + x] = r;
            pixel_values[output_plane + static_cast<std::size_t>(y) * out_w + x] = g;
            pixel_values[2U * output_plane + static_cast<std::size_t>(y) * out_w + x] = b;
        }
    }
    return pixel_values;
}

} // namespace trtmc
