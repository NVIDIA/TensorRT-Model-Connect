/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "media.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::hunyuan_vl {
namespace {
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

// PyTorch's uint8 antialiased bicubic: separable passes with int16 fixed-point coefficients.
AntialiasWeights bicubic_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    AntialiasWeights result;
    result.stride = kernel_size;
    while (result.stride % static_cast<int32_t>(sizeof(int32_t)) != 0)
        ++result.stride;
    result.starts.resize(static_cast<size_t>(output_size));
    result.sizes.resize(static_cast<size_t>(output_size));
    result.values.assign(static_cast<size_t>(output_size) * result.stride, 0);
    std::vector<double> floating(static_cast<size_t>(output_size) * kernel_size, 0.0);
    double maximum_weight = 0.0;
    for (int32_t out = 0; out < output_size; ++out) {
        const double center = scale * (out + 0.5);
        const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int32_t start = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t size =
            std::clamp(std::min(static_cast<int32_t>(center + support + 0.5), input_size) - start,
                       0, kernel_size);
        result.starts[static_cast<size_t>(out)] = start;
        result.sizes[static_cast<size_t>(out)] = size;
        double total = 0.0;
        double* row = floating.data() + static_cast<size_t>(out) * kernel_size;
        for (int32_t i = 0; i < size; ++i) {
            row[i] = keys_cubic((i + start - center + 0.5) * inverse_scale);
            total += row[i];
        }
        if (total != 0.0)
            for (int32_t i = 0; i < size; ++i) {
                row[i] /= total;
                maximum_weight = std::max(maximum_weight, row[i]);
            }
    }
    // The greatest fixed-point precision whose largest coefficient still fits int16.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t out = 0; out < output_size; ++out)
        for (int32_t i = 0; i < kernel_size; ++i) {
            const double value = floating[static_cast<size_t>(out) * kernel_size + i] * multiplier;
            result.values[static_cast<size_t>(out) * result.stride + i] =
                static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                 : static_cast<int32_t>(value + 0.5));
        }
    return result;
}

uint8_t fixed_point(const uint8_t* source, int32_t stride, const int16_t* weights, int32_t count,
                    uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t i = 0; i < count; ++i)
        value += static_cast<int32_t>(source[static_cast<size_t>(i) * stride]) * weights[i];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

std::vector<unsigned char> horizontal(const unsigned char* raw, int32_t width, int32_t height,
                                      int32_t target_width) {
    const auto weights = bicubic_weights(width, target_width);
    std::vector<unsigned char> resized(static_cast<size_t>(height) * target_width * 3);
    for (int32_t y = 0; y < height; ++y)
        for (int32_t x = 0; x < target_width; ++x) {
            const int32_t start = weights.starts[static_cast<size_t>(x)];
            const int16_t* coefficients =
                weights.values.data() + static_cast<size_t>(x) * weights.stride;
            for (int32_t c = 0; c < 3; ++c)
                resized[(static_cast<size_t>(y) * target_width + x) * 3 + c] = fixed_point(
                    raw + (static_cast<size_t>(y) * width + start) * 3 + c, 3, coefficients,
                    weights.sizes[static_cast<size_t>(x)], weights.precision);
        }
    return resized;
}

std::vector<unsigned char> vertical(const unsigned char* raw, int32_t width, int32_t height,
                                    int32_t target_height) {
    const auto weights = bicubic_weights(height, target_height);
    std::vector<unsigned char> resized(static_cast<size_t>(target_height) * width * 3);
    for (int32_t y = 0; y < target_height; ++y) {
        const int32_t start = weights.starts[static_cast<size_t>(y)];
        const int16_t* coefficients =
            weights.values.data() + static_cast<size_t>(y) * weights.stride;
        for (int32_t x = 0; x < width; ++x)
            for (int32_t c = 0; c < 3; ++c)
                resized[(static_cast<size_t>(y) * width + x) * 3 + c] = fixed_point(
                    raw + (static_cast<size_t>(start) * width + x) * 3 + c, width * 3, coefficients,
                    weights.sizes[static_cast<size_t>(y)], weights.precision);
    }
    return resized;
}
} // namespace

std::vector<unsigned char> resize_bicubic_antialias_u8(const unsigned char* raw, int32_t width,
                                                       int32_t height, int32_t target_width,
                                                       int32_t target_height) {
    if (!raw || width <= 0 || height <= 0 || target_width <= 0 || target_height <= 0)
        throw std::invalid_argument("Invalid HunYuanVL resize dimensions");
    if (width == target_width && height == target_height)
        return {raw, raw + static_cast<size_t>(width) * height * 3};
    std::vector<unsigned char> wide;
    const unsigned char* source = raw;
    if (width != target_width) {
        wide = horizontal(raw, width, height, target_width);
        source = wide.data();
    }
    if (height == target_height)
        return wide;
    return vertical(source, target_width, height, target_height);
}

ImagePlan plan_image(internal::ImageView image, const ImageSettings& settings, int max_patches) {
    if (!image.data || image.height == 0 || image.width == 0 || image.channels != 3 ||
        image.height > 100000 || image.width > 100000)
        throw std::invalid_argument("HunYuanVL requires nonempty RGB images");
    if (image.format != internal::ImageFormat::UInt8 &&
        image.format != internal::ImageFormat::Float32)
        throw std::invalid_argument("HunYuanVL requires uint8 or float32 pixels");
    const size_t element = image.format == internal::ImageFormat::UInt8 ? 1 : 4;
    if (image.byte_size != size_t(image.height) * image.width * 3 * element)
        throw std::invalid_argument("HunYuanVL image byte count does not match its dimensions");
    const double height = image.height, width = image.width;
    if (std::max(height, width) / std::min(height, width) > 200)
        throw std::invalid_argument("HunYuanVL image aspect ratio must not exceed 200");
    // The reference smart_resize: round half to even, then bound the area.
    const int factor = settings.patch * settings.merge;
    int64_t h = int64_t(std::nearbyint(height / factor)) * factor;
    int64_t w = int64_t(std::nearbyint(width / factor)) * factor;
    if (h * w > settings.max_pixels) {
        const double beta = std::sqrt(height * width / double(settings.max_pixels));
        h = std::max<int64_t>(factor, int64_t(std::floor(height / beta / factor)) * factor);
        w = std::max<int64_t>(factor, int64_t(std::floor(width / beta / factor)) * factor);
    } else if (h * w < settings.min_pixels) {
        const double beta = std::sqrt(double(settings.min_pixels) / (height * width));
        h = int64_t(std::ceil(height * beta / factor)) * factor;
        w = int64_t(std::ceil(width * beta / factor)) * factor;
    }
    if ((h / settings.patch) * (w / settings.patch) > max_patches)
        throw std::invalid_argument(
            "HunYuanVL image needs more vision patches than this bundle was built for");
    return {image, int(h), int(w), settings.patch, settings.merge};
}

std::vector<float> patchify(const ImagePlan& plan, const ImageSettings& settings) {
    const auto& image = plan.image;
    std::vector<uint8_t> converted;
    const auto* source = static_cast<const uint8_t*>(image.data);
    if (image.format == internal::ImageFormat::Float32) {
        const size_t count = size_t(image.height) * image.width * 3;
        const auto* values = static_cast<const float*>(image.data);
        converted.resize(count);
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]) || values[i] < 0 || values[i] > 1)
                throw std::invalid_argument("HunYuanVL float RGB pixels must be finite in [0,1]");
            converted[i] = static_cast<uint8_t>(std::nearbyint(values[i] * 255.0F));
        }
        source = converted.data();
    }
    const auto pixels = resize_bicubic_antialias_u8(source, int32_t(image.width),
                                                    int32_t(image.height), plan.width, plan.height);
    // The reference folds rescaling into normalization: (x - mean / r) / (std / r).
    std::array<float, 3> mean{}, std{};
    for (int c = 0; c < 3; ++c) {
        mean[c] = float(settings.mean[c] * (1.0 / settings.rescale_factor));
        std[c] = float(settings.std[c] * (1.0 / settings.rescale_factor));
    }
    const int p = plan.patch, m = plan.merge;
    std::vector<float> result;
    result.reserve(size_t(plan.height) * plan.width * 3);
    for (int by = 0; by < plan.rows(); by += m)
        for (int bx = 0; bx < plan.cols(); bx += m)
            for (int dy = 0; dy < m; ++dy)
                for (int dx = 0; dx < m; ++dx)
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < p; ++y)
                            for (int x = 0; x < p; ++x) {
                                const size_t at = (size_t((by + dy) * p + y) * plan.width +
                                                   size_t((bx + dx) * p + x)) *
                                                      3 +
                                                  size_t(c);
                                result.push_back((float(pixels[at]) - mean[c]) / std[c]);
                            }
    return result;
}

void position_corners(const ImagePlan& plan, int edge, std::vector<int32_t>& ids,
                      std::vector<float>& lambdas) {
    if (edge <= 1)
        throw std::invalid_argument("Invalid HunYuanVL position grid");
    struct Axis {
        std::vector<int32_t> low, high;
        std::vector<float> keep, take;
    };
    // The reference's bilinear source index (align_corners=false): FP32, with its multiply-add
    // fused, as PyTorch evaluates it; a separate rounding moves large indices by one ulp.
    const auto axis = [edge](int size) {
        Axis result;
        const float scale = float(edge) / float(size);
        for (int i = 0; i < size; ++i) {
            float source = std::fma(scale, float(i) + 0.5F, -0.5F);
            source = source < 0 ? 0 : source;
            const int low = int(source);
            const float take = source - float(low);
            result.low.push_back(low);
            result.high.push_back(low + (low < edge - 1 ? 1 : 0));
            result.take.push_back(take);
            result.keep.push_back(1.0F - take);
        }
        return result;
    };
    const auto ys = axis(plan.rows()), xs = axis(plan.cols());
    ids.clear();
    lambdas.clear();
    const int m = plan.merge;
    for (int by = 0; by < plan.rows(); by += m)
        for (int bx = 0; bx < plan.cols(); bx += m)
            for (int dy = 0; dy < m; ++dy)
                for (int dx = 0; dx < m; ++dx) {
                    const int r = by + dy, c = bx + dx;
                    // Row 0 of the table is the class token; the patch grid starts at row 1.
                    for (const int y : {ys.low[r], ys.high[r]})
                        for (const int x : {xs.low[c], xs.high[c]})
                            ids.push_back(1 + y * edge + x);
                    lambdas.insert(lambdas.end(), {ys.keep[r], ys.take[r], xs.keep[c], xs.take[c]});
                }
}
} // namespace trtmc::hunyuan_vl
