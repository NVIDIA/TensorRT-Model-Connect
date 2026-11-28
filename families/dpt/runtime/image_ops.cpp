/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dpt/runtime/image_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::dpt {
namespace {

// Pillow keeps 8 bits for the result and two for overflow of an int.
constexpr int kPrecisionBits = 32 - 8 - 2;

void require_size(std::int32_t height, std::int32_t width, const char* what) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument(std::string("dpt ") + what + " dimensions must be positive");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 4U;
    if (static_cast<std::size_t>(height) * static_cast<std::size_t>(width) > limit)
        throw std::invalid_argument(std::string("dpt ") + what + " is too large");
}

std::uint8_t clip8(int value) {
    const int shifted = value >> kPrecisionBits;
    return static_cast<std::uint8_t>(std::clamp(shifted, 0, 255));
}

double bilinear_filter(double x) {
    if (x < 0.0)
        x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

// One axis of Pillow's resampling: for every output index the first input
// index, the tap count and fixed-point weights (precompute_coeffs with the
// bilinear filter over the box (0, in_size), then normalize_coeffs_8bpc).
struct AxisCoefficients {
    int ksize{0};
    std::vector<int> first;
    std::vector<int> count;
    std::vector<std::int32_t> weights;
};

AxisCoefficients precompute_coefficients(int in_size, int out_size) {
    const float in0 = 0.0F;
    const auto in1 = static_cast<float>(in_size);
    const double scale = static_cast<double>(in1 - in0) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 1.0 * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    const double inv_filterscale = 1.0 / filterscale;

    AxisCoefficients result;
    result.ksize = ksize;
    result.first.resize(static_cast<std::size_t>(out_size));
    result.count.resize(static_cast<std::size_t>(out_size));
    result.weights.assign(static_cast<std::size_t>(out_size) * static_cast<std::size_t>(ksize), 0);
    std::vector<double> kernel(static_cast<std::size_t>(ksize));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0)
            xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size)
            xmax = in_size;
        xmax -= xmin;
        double total = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double weight = bilinear_filter((x + xmin - center + 0.5) * inv_filterscale);
            kernel[static_cast<std::size_t>(x)] = weight;
            total += weight;
        }
        if (total != 0.0) {
            for (int x = 0; x < xmax; ++x)
                kernel[static_cast<std::size_t>(x)] /= total;
        }
        std::int32_t* fixed = &result.weights[static_cast<std::size_t>(xx) * ksize];
        for (int x = 0; x < xmax; ++x) {
            const double value = kernel[static_cast<std::size_t>(x)];
            const double scaled = value * (1 << kPrecisionBits);
            fixed[x] = value < 0 ? static_cast<int>(-0.5 + scaled) : static_cast<int>(0.5 + scaled);
        }
        result.first[static_cast<std::size_t>(xx)] = xmin;
        result.count[static_cast<std::size_t>(xx)] = xmax;
    }
    return result;
}

std::vector<std::uint8_t> horizontal_pass(const std::vector<std::uint8_t>& source, int rows,
                                          int in_width, int out_width) {
    const AxisCoefficients axis = precompute_coefficients(in_width, out_width);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(rows) * out_width * 3U);
    for (int y = 0; y < rows; ++y) {
        const std::uint8_t* line = &source[static_cast<std::size_t>(y) * in_width * 3U];
        std::uint8_t* target = &output[static_cast<std::size_t>(y) * out_width * 3U];
        for (int xx = 0; xx < out_width; ++xx) {
            const int xmin = axis.first[static_cast<std::size_t>(xx)];
            const int taps = axis.count[static_cast<std::size_t>(xx)];
            const std::int32_t* weight = &axis.weights[static_cast<std::size_t>(xx) * axis.ksize];
            int sums[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1),
                           1 << (kPrecisionBits - 1)};
            for (int x = 0; x < taps; ++x) {
                const std::uint8_t* pixel = line + static_cast<std::size_t>(x + xmin) * 3U;
                for (int channel = 0; channel < 3; ++channel)
                    sums[channel] += pixel[channel] * weight[x];
            }
            for (int channel = 0; channel < 3; ++channel)
                target[static_cast<std::size_t>(xx) * 3U + channel] = clip8(sums[channel]);
        }
    }
    return output;
}

std::vector<std::uint8_t> vertical_pass(const std::vector<std::uint8_t>& source, int width,
                                        int in_height, int out_height) {
    const AxisCoefficients axis = precompute_coefficients(in_height, out_height);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(out_height) * width * 3U);
    for (int yy = 0; yy < out_height; ++yy) {
        const int ymin = axis.first[static_cast<std::size_t>(yy)];
        const int taps = axis.count[static_cast<std::size_t>(yy)];
        const std::int32_t* weight = &axis.weights[static_cast<std::size_t>(yy) * axis.ksize];
        std::uint8_t* target = &output[static_cast<std::size_t>(yy) * width * 3U];
        for (int xx = 0; xx < width; ++xx) {
            int sums[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1),
                           1 << (kPrecisionBits - 1)};
            for (int y = 0; y < taps; ++y) {
                const std::uint8_t* pixel =
                    &source[(static_cast<std::size_t>(y + ymin) * width + xx) * 3U];
                for (int channel = 0; channel < 3; ++channel)
                    sums[channel] += pixel[channel] * weight[y];
            }
            for (int channel = 0; channel < 3; ++channel)
                target[static_cast<std::size_t>(xx) * 3U + channel] = clip8(sums[channel]);
        }
    }
    return output;
}

// PyTorch's cubic convolution kernel (A = -0.75) in float, as
// get_cubic_upsample_coefficients evaluates it.
float cubic_convolution1(float x, float a) {
    return ((a + 2) * x - (a + 3)) * x * x + 1;
}

float cubic_convolution2(float x, float a) {
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}

struct CubicAxis {
    std::vector<std::array<std::int32_t, 4>> index;
    std::vector<std::array<float, 4>> weight;
};

CubicAxis cubic_axis(std::int32_t in_size, std::int32_t out_size) {
    // area_pixel_compute_scale and area_pixel_compute_source_index with
    // align_corners=False and no explicit scale, in float.
    const float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
    CubicAxis axis;
    axis.index.resize(static_cast<std::size_t>(out_size));
    axis.weight.resize(static_cast<std::size_t>(out_size));
    for (std::int32_t i = 0; i < out_size; ++i) {
        const float real = scale * (static_cast<float>(i) + 0.5F) - 0.5F;
        const auto floor_index = static_cast<std::int64_t>(std::floor(real));
        const std::int64_t input_index = std::min<std::int64_t>(floor_index, in_size - 1);
        const float lambda = std::min(std::max(real - static_cast<float>(input_index), 0.0F), 1.0F);
        const float a = -0.75F;
        const float x1 = lambda;
        const float x2 = static_cast<float>(1.0 - static_cast<double>(lambda));
        auto& weight = axis.weight[static_cast<std::size_t>(i)];
        weight[0] = cubic_convolution2(static_cast<float>(static_cast<double>(x1) + 1.0), a);
        weight[1] = cubic_convolution1(x1, a);
        weight[2] = cubic_convolution1(x2, a);
        weight[3] = cubic_convolution2(static_cast<float>(static_cast<double>(x2) + 1.0), a);
        auto& index = axis.index[static_cast<std::size_t>(i)];
        for (std::int64_t tap = 0; tap < 4; ++tap) {
            index[static_cast<std::size_t>(tap)] = static_cast<std::int32_t>(
                std::clamp<std::int64_t>(input_index + tap - 1, 0, in_size - 1));
        }
    }
    return axis;
}

// rescale() multiplies in float64 and stores float32; normalize() subtracts
// and divides in float32. The caller has checked that the product fits float32.
float rescaled(std::uint8_t level, double rescale_factor) {
    return static_cast<float>(static_cast<double>(level) * rescale_factor);
}

} // namespace

void check_preprocess_config(const PreprocessConfig& config) {
    require_size(config.input_image_h, config.input_image_w, "engine input");
    constexpr double kFloatMax = std::numeric_limits<float>::max();
    if (!std::isfinite(config.rescale_factor) || config.rescale_factor <= 0.0 ||
        config.rescale_factor > kFloatMax / 255.0) {
        throw std::invalid_argument(
            "dpt rescale factor must be positive and keep 255 within float32");
    }
    for (std::size_t channel = 0; channel < 3U; ++channel) {
        const float mean = config.image_mean[channel];
        const float std = config.image_std[channel];
        if (!std::isfinite(mean) || !std::isfinite(std) || std == 0.0F)
            throw std::invalid_argument("dpt image mean and std must be finite, std non-zero");
        for (int level = 0; level < 256; ++level) {
            const float value = rescaled(static_cast<std::uint8_t>(level), config.rescale_factor);
            if (!std::isfinite((value - mean) / std)) {
                throw std::invalid_argument(
                    "dpt image mean and std overflow float32 for an 8-bit sample");
            }
        }
    }
}

std::vector<std::uint8_t> to_rgb8(const float* pixels, std::int32_t height, std::int32_t width) {
    if (pixels == nullptr)
        throw std::invalid_argument("dpt source image must be non-empty");
    require_size(height, width, "source image");
    const auto count = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3U;
    std::vector<std::uint8_t> rgb(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float value = pixels[index];
        if (!std::isfinite(value))
            throw std::invalid_argument("dpt source image contains a non-finite pixel");
        const float level = std::round(value * 255.0F);
        rgb[index] = static_cast<std::uint8_t>(std::clamp(level, 0.0F, 255.0F));
    }
    return rgb;
}

std::vector<std::uint8_t> resize_bilinear_rgb8(const std::uint8_t* rgb, std::int32_t height,
                                               std::int32_t width, std::int32_t out_height,
                                               std::int32_t out_width) {
    if (rgb == nullptr)
        throw std::invalid_argument("dpt resize source must be non-empty");
    require_size(height, width, "resize source");
    require_size(out_height, out_width, "resize target");

    const bool need_horizontal = out_width != width;
    const bool need_vertical = out_height != height;
    // Pillow 13 runs the vertical pass first when the height shrinks by more
    // than twice as many pixels as the width does.
    const bool horizontal_first =
        !((height - out_height) > 0 && static_cast<std::int64_t>(height - out_height) >
                                           static_cast<std::int64_t>(width - out_width) * 2);

    std::vector<std::uint8_t> image(rgb, rgb + static_cast<std::size_t>(height) * width * 3U);
    int rows = height;
    int columns = width;
    const auto horizontal = [&]() {
        image = horizontal_pass(image, rows, columns, out_width);
        columns = out_width;
    };
    const auto vertical = [&]() {
        image = vertical_pass(image, columns, rows, out_height);
        rows = out_height;
    };
    if (horizontal_first) {
        if (need_horizontal)
            horizontal();
        if (need_vertical)
            vertical();
    } else {
        if (need_vertical)
            vertical();
        if (need_horizontal)
            horizontal();
    }
    return image;
}

std::vector<float> preprocess_image(const float* pixels, std::int32_t height, std::int32_t width,
                                    const PreprocessConfig& config) {
    check_preprocess_config(config);
    const auto rgb = to_rgb8(pixels, height, width);
    const auto resized =
        resize_bilinear_rgb8(rgb.data(), height, width, config.input_image_h, config.input_image_w);

    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::size_t index = 0; index < plane; ++index) {
        for (std::size_t channel = 0; channel < 3U; ++channel) {
            const float value = rescaled(resized[index * 3U + channel], config.rescale_factor);
            pixel_values[channel * plane + index] =
                (value - config.image_mean[channel]) / config.image_std[channel];
        }
    }
    return pixel_values;
}

std::vector<float> resize_bicubic(const float* source, std::int32_t source_h, std::int32_t source_w,
                                  std::int32_t target_h, std::int32_t target_w) {
    if (source == nullptr)
        throw std::invalid_argument("dpt depth map must be non-empty");
    require_size(source_h, source_w, "depth map");
    require_size(target_h, target_w, "depth target");
    const CubicAxis rows = cubic_axis(source_h, target_h);
    const CubicAxis columns = cubic_axis(source_w, target_w);

    std::vector<float> output(static_cast<std::size_t>(target_h) * target_w);
    for (std::int32_t y = 0; y < target_h; ++y) {
        const auto& row_index = rows.index[static_cast<std::size_t>(y)];
        const auto& row_weight = rows.weight[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < target_w; ++x) {
            const auto& column_index = columns.index[static_cast<std::size_t>(x)];
            const auto& column_weight = columns.weight[static_cast<std::size_t>(x)];
            // Width inside, height outside, each accumulated tap by tap.
            float value = 0.0F;
            for (std::size_t j = 0; j < 4U; ++j) {
                const float* line = source + static_cast<std::size_t>(row_index[j]) * source_w;
                float across = line[column_index[0]] * column_weight[0];
                for (std::size_t k = 1; k < 4U; ++k)
                    across += line[column_index[k]] * column_weight[k];
                value = j == 0 ? across * row_weight[0] : value + across * row_weight[j];
            }
            output[static_cast<std::size_t>(y) * target_w + x] = value;
        }
    }
    return output;
}

} // namespace trtmc::dpt
