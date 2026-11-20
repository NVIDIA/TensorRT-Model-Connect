/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "media.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace trtmc::cosmos3_edge {
template <class F>
void parallel_for_ranges(int32_t count, int32_t, F&& f) {
    f(0, count);
}
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

static double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static int32_t aligned_coefficient_stride(int32_t kernel_size) {
    int32_t stride = kernel_size;
    while (stride % static_cast<int32_t>(sizeof(int32_t)) != 0)
        ++stride;
    return stride;
}

static double build_floating_antialias_weights(int32_t input_size, int32_t output_size,
                                               double scale, double support, int32_t kernel_size,
                                               AntialiasWeights& result,
                                               std::vector<double>& floating) {
    double maximum_weight = 0.0;
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = scale * (output_index + 0.5);
        const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int32_t start = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t size =
            std::clamp(std::min(static_cast<int32_t>(center + support + 0.5), input_size) - start,
                       0, kernel_size);
        result.starts[static_cast<std::size_t>(output_index)] = start;
        result.sizes[static_cast<std::size_t>(output_index)] = size;

        double total = 0.0;
        double* row = floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        for (int32_t index = 0; index < size; ++index) {
            row[index] = keys_cubic((index + start - center + 0.5) * inverse_scale);
            total += row[index];
        }
        if (total != 0.0) {
            for (int32_t index = 0; index < size; ++index) {
                row[index] /= total;
                maximum_weight = std::max(maximum_weight, row[index]);
            }
        }
    }
    return maximum_weight;
}

static void quantize_antialias_weights(const std::vector<double>& floating, int32_t output_size,
                                       int32_t kernel_size, double maximum_weight,
                                       AntialiasWeights& result) {
    // Select the greatest fixed-point precision whose largest coefficient fits
    // in int16, exactly as PyTorch's uint8 antialias implementation does.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }

    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double* source =
            floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        int16_t* destination =
            result.values.data() + static_cast<std::size_t>(output_index) * result.stride;
        for (int32_t index = 0; index < kernel_size; ++index) {
            const double value = source[index] * multiplier;
            destination[index] =
                static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                 : static_cast<int32_t>(value + 0.5));
        }
    }
}

static AntialiasWeights make_bicubic_antialias_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

    // PyTorch pads each int16 coefficient row to a 32-bit-aligned size in its
    // optimized uint8 path. The padding does not participate in convolution.
    AntialiasWeights result;
    result.stride = aligned_coefficient_stride(kernel_size);
    result.starts.resize(static_cast<std::size_t>(output_size));
    result.sizes.resize(static_cast<std::size_t>(output_size));
    result.values.assign(static_cast<std::size_t>(output_size) * result.stride, 0);

    std::vector<double> floating(static_cast<std::size_t>(output_size) * kernel_size, 0.0);
    const double maximum_weight = build_floating_antialias_weights(
        input_size, output_size, scale, support, kernel_size, result, floating);
    quantize_antialias_weights(floating, output_size, kernel_size, maximum_weight, result);
    return result;
}

static uint8_t fixed_point_pixel(const uint8_t* source, int32_t source_stride,
                                 const int16_t* weights, int32_t count, uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t index = 0; index < count; ++index)
        value += static_cast<int32_t>(source[static_cast<std::size_t>(index) * source_stride]) *
                 weights[index];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

static bool valid_resize_dimensions(const unsigned char* raw, int32_t width, int32_t height,
                                    int32_t target_width, int32_t target_height) {
    return raw != nullptr && width > 0 && height > 0 && target_width > 0 && target_height > 0;
}

static std::vector<unsigned char> resize_bicubic_horizontal(const unsigned char* raw, int32_t width,
                                                            int32_t height, int32_t target_width) {
    const auto weights = make_bicubic_antialias_weights(width, target_width);
    std::vector<unsigned char> resized(static_cast<std::size_t>(height) * target_width * 3);
    parallel_for_ranges(height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            for (int32_t x = 0; x < target_width; ++x) {
                const int32_t start = weights.starts[static_cast<std::size_t>(x)];
                const int32_t count = weights.sizes[static_cast<std::size_t>(x)];
                const int16_t* coefficients =
                    weights.values.data() + static_cast<std::size_t>(x) * weights.stride;
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(y) * width + start) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, 3, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

static std::vector<unsigned char> resize_bicubic_vertical(const unsigned char* raw, int32_t width,
                                                          int32_t height, int32_t target_height) {
    const auto weights = make_bicubic_antialias_weights(height, target_height);
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_height) * width * 3);
    const int32_t row_stride = width * 3;
    parallel_for_ranges(target_height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            const int32_t start = weights.starts[static_cast<std::size_t>(y)];
            const int32_t count = weights.sizes[static_cast<std::size_t>(y)];
            const int16_t* coefficients =
                weights.values.data() + static_cast<std::size_t>(y) * weights.stride;
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(start) * width + x) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, row_stride, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

std::vector<unsigned char> resize_bicubic_antialias_u8(const unsigned char* raw, int32_t width,
                                                       int32_t height, int32_t target_width,
                                                       int32_t target_height) {
    if (!valid_resize_dimensions(raw, width, height, target_width, target_height))
        return {};
    if (width == target_width && height == target_height) {
        return {raw, raw + static_cast<std::size_t>(width) * height * 3};
    }

    std::vector<unsigned char> horizontal;
    const unsigned char* vertical_source = raw;
    if (width != target_width) {
        horizontal = resize_bicubic_horizontal(raw, width, height, target_width);
        vertical_source = horizontal.data();
    }

    if (height == target_height)
        return horizontal;

    return resize_bicubic_vertical(vertical_source, target_width, height, target_height);
}

Frame plan_frame(internal::ImageView image, int patch, int merge, int frames, int min_pixels,
                 int max_pixels, int max_patches) {
    if (!image.data || image.height == 0 || image.width == 0 || image.channels != 3 ||
        image.height > 100000 || image.width > 100000 || patch <= 0 || merge <= 0 || frames <= 0)
        throw std::invalid_argument("Cosmos3-Edge requires nonempty RGB frames");
    const size_t bytes_per_pixel = image.format == internal::ImageFormat::UInt8 ? 1 : 4;
    if (image.format != internal::ImageFormat::UInt8 &&
        image.format != internal::ImageFormat::Float32)
        throw std::invalid_argument("Cosmos3-Edge requires uint8 or float32 pixels");
    if (image.byte_size != size_t(image.height) * image.width * 3 * bytes_per_pixel)
        throw std::invalid_argument("Cosmos3-Edge image byte count does not match its dimensions");
    const int factor = patch * merge;
    int height = image.height, width = image.width;
    if (std::min(height, width) < factor) {
        const double scale = double(factor) / std::min(height, width);
        height = int(height * scale);
        width = int(width * scale);
    }
    if (double(std::max(height, width)) / std::min(height, width) > 200)
        throw std::invalid_argument("Cosmos3-Edge frame aspect ratio exceeds 200");
    int h = int(std::nearbyint(double(height) / factor)) * factor;
    int w = int(std::nearbyint(double(width) / factor)) * factor;
    if (double(frames) * h * w > max_pixels) {
        const double beta = std::sqrt(double(frames) * height * width / max_pixels);
        h = std::max(factor, int(std::floor(height / beta / factor)) * factor);
        w = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    } else if (double(frames) * h * w < min_pixels) {
        const double beta = std::sqrt(double(min_pixels) / (double(frames) * height * width));
        h = int(std::ceil(height * beta / factor)) * factor;
        w = int(std::ceil(width * beta / factor)) * factor;
    }
    if (int64_t(h / patch) * (w / patch) > max_patches)
        throw std::invalid_argument("Cosmos3-Edge frame exceeds the built visual patch capacity");
    return {image, h, w, patch, merge};
}

std::vector<float> patchify(const Frame& frame) {
    const auto& image = frame.image;
    std::vector<uint8_t> converted;
    const auto* source = static_cast<const uint8_t*>(image.data);
    if (image.format == internal::ImageFormat::Float32) {
        const size_t count = size_t(image.height) * image.width * 3;
        const auto* values = static_cast<const float*>(image.data);
        converted.resize(count);
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]) || values[i] < 0 || values[i] > 1)
                throw std::invalid_argument(
                    "Cosmos3-Edge float RGB pixels must be finite in [0,1]");
            converted[i] = static_cast<uint8_t>(std::nearbyint(values[i] * 255.0F));
        }
        source = converted.data();
    }
    const auto pixels =
        resize_bicubic_antialias_u8(source, image.width, image.height, frame.width, frame.height);
    std::vector<float> result;
    result.reserve(size_t(frame.height) * frame.width * 3);
    for (int by = 0; by < frame.rows(); by += frame.merge)
        for (int bx = 0; bx < frame.cols(); bx += frame.merge)
            for (int dy = 0; dy < frame.merge; ++dy)
                for (int dx = 0; dx < frame.merge; ++dx)
                    for (int y = 0; y < frame.patch; ++y)
                        for (int x = 0; x < frame.patch; ++x)
                            for (int c = 0; c < 3; ++c) {
                                const size_t at =
                                    (size_t((by + dy) * frame.patch + y) * frame.width +
                                     (bx + dx) * frame.patch + x) *
                                        3 +
                                    c;
                                result.push_back((pixels[at] / 255.0F - 0.5F) / 0.5F);
                            }
    return result;
}

void interpolate_positions(const Frame& frame, int table_side, std::vector<float>& weights) {
    if (table_side <= 0)
        throw std::invalid_argument("Invalid Cosmos3-Edge position table");
    const auto axis = [table_side](int output_size) {
        std::vector<float> result(size_t(output_size) * table_side);
        const double scale = double(table_side) / output_size;
        const double support = std::max(scale, 1.0);
        for (int out = 0; out < output_size; ++out) {
            const double center = (out + 0.5) * scale;
            double sum = 0;
            for (int source = 0; source < table_side; ++source) {
                const float value =
                    float(std::max(0.0, 1.0 - std::abs((source + 0.5 - center) / support)));
                result[size_t(out) * table_side + source] = value;
                sum += value;
            }
            for (int source = 0; source < table_side; ++source)
                result[size_t(out) * table_side + source] /= float(sum);
        }
        return result;
    };
    const auto ys = axis(frame.rows()), xs = axis(frame.cols());
    const int table_size = table_side * table_side;
    weights.resize(size_t(frame.rows()) * frame.cols() * table_size);
    int row = 0;
    for (int by = 0; by < frame.rows(); by += frame.merge)
        for (int bx = 0; bx < frame.cols(); bx += frame.merge)
            for (int dy = 0; dy < frame.merge; ++dy)
                for (int dx = 0; dx < frame.merge; ++dx, ++row)
                    for (int y = 0; y < table_side; ++y)
                        for (int x = 0; x < table_side; ++x)
                            weights[size_t(row) * table_size + y * table_side + x] =
                                ys[size_t(by + dy) * table_side + y] *
                                xs[size_t(bx + dx) * table_side + x];
}
} // namespace trtmc::cosmos3_edge
