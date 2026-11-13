/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/owlv2/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

void validate(const Owlv2PreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("owlv2 target dimensions must be positive");
    if (config.input_image_h != config.input_image_w)
        throw std::invalid_argument("owlv2 pads to a square, so both sides must match");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("owlv2 image statistics must have three channels");
    for (const float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("owlv2 image std must be non-zero");
    }
}

// Reflect without repeating the edge sample: abcd -> dcb|abcd|cba, which is
// what torch's "reflect" padding does and what the reference blur runs under.
std::int32_t reflect(std::int32_t index, std::int32_t size) {
    if (size == 1)
        return 0;
    while (index < 0 || index >= size) {
        if (index < 0)
            index = -index;
        if (index >= size)
            index = 2 * (size - 1) - index;
    }
    return index;
}

std::vector<float> gaussian_kernel(float sigma, std::int32_t size) {
    std::vector<float> kernel(static_cast<std::size_t>(size));
    const double half = (static_cast<double>(size) - 1.0) * 0.5;
    double total = 0.0;
    for (std::int32_t i = 0; i < size; ++i) {
        const double x = static_cast<double>(i) - half;
        const double value = std::exp(-0.5 * (x / sigma) * (x / sigma));
        kernel[static_cast<std::size_t>(i)] = static_cast<float>(value);
        total += value;
    }
    for (float& value : kernel)
        value = static_cast<float>(static_cast<double>(value) / total);
    return kernel;
}

// Separable blur over an interleaved RGB square.
std::vector<float> blur(const std::vector<float>& source, std::int32_t side, float sigma) {
    const auto radius = static_cast<std::int32_t>(std::ceil(3.0F * sigma));
    const std::int32_t size = 2 * radius + 1;
    const auto kernel = gaussian_kernel(sigma, size);
    std::vector<float> horizontal(source.size());
    for (std::int32_t y = 0; y < side; ++y) {
        for (std::int32_t x = 0; x < side; ++x) {
            for (std::int32_t c = 0; c < 3; ++c) {
                double total = 0.0;
                for (std::int32_t k = 0; k < size; ++k) {
                    const std::int32_t sx = reflect(x + k - radius, side);
                    total += static_cast<double>(kernel[static_cast<std::size_t>(k)]) *
                             source[static_cast<std::size_t>((y * side + sx) * 3 + c)];
                }
                horizontal[static_cast<std::size_t>((y * side + x) * 3 + c)] =
                    static_cast<float>(total);
            }
        }
    }
    std::vector<float> output(source.size());
    for (std::int32_t y = 0; y < side; ++y) {
        for (std::int32_t x = 0; x < side; ++x) {
            for (std::int32_t c = 0; c < 3; ++c) {
                double total = 0.0;
                for (std::int32_t k = 0; k < size; ++k) {
                    const std::int32_t sy = reflect(y + k - radius, side);
                    total += static_cast<double>(kernel[static_cast<std::size_t>(k)]) *
                             horizontal[static_cast<std::size_t>((sy * side + x) * 3 + c)];
                }
                output[static_cast<std::size_t>((y * side + x) * 3 + c)] =
                    static_cast<float>(total);
            }
        }
    }
    return output;
}

// Bilinear with align_corners=false, the sampling torch resizes with once the
// anti-aliasing filter has already been applied separately.
std::vector<float> resize_bilinear(const std::vector<float>& source, std::int32_t side,
                                   std::int32_t target) {
    const double scale = static_cast<double>(side) / static_cast<double>(target);
    std::vector<std::int32_t> low(static_cast<std::size_t>(target));
    std::vector<std::int32_t> high(static_cast<std::size_t>(target));
    std::vector<double> fraction(static_cast<std::size_t>(target));
    for (std::int32_t i = 0; i < target; ++i) {
        double centre = scale * (static_cast<double>(i) + 0.5) - 0.5;
        if (centre < 0.0)
            centre = 0.0;
        auto first = static_cast<std::int32_t>(std::floor(centre));
        first = std::min(first, side - 1);
        low[static_cast<std::size_t>(i)] = first;
        high[static_cast<std::size_t>(i)] = std::min(first + 1, side - 1);
        fraction[static_cast<std::size_t>(i)] = centre - static_cast<double>(first);
    }
    std::vector<float> output(static_cast<std::size_t>(target) * target * 3);
    for (std::int32_t y = 0; y < target; ++y) {
        const auto y0 = low[static_cast<std::size_t>(y)];
        const auto y1 = high[static_cast<std::size_t>(y)];
        const double wy = fraction[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < target; ++x) {
            const auto x0 = low[static_cast<std::size_t>(x)];
            const auto x1 = high[static_cast<std::size_t>(x)];
            const double wx = fraction[static_cast<std::size_t>(x)];
            for (std::int32_t c = 0; c < 3; ++c) {
                const auto at = [&](std::int32_t sy, std::int32_t sx) {
                    return static_cast<double>(
                        source[static_cast<std::size_t>((sy * side + sx) * 3 + c)]);
                };
                const double top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                const double bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                output[static_cast<std::size_t>((y * target + x) * 3 + c)] =
                    static_cast<float>(top * (1.0 - wy) + bottom * wy);
            }
        }
    }
    return output;
}

} // namespace

std::int32_t compute_owlv2_pad_side(std::int32_t image_height, std::int32_t image_width) {
    if (image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("owlv2 source dimensions must be positive");
    return std::max(image_height, image_width);
}

float compute_owlv2_antialias_sigma(std::int32_t pad_side, std::int32_t target) {
    if (pad_side <= 0 || target <= 0)
        throw std::invalid_argument("owlv2 resize dimensions must be positive");
    // sigma = (factor - 1) / 2, clamped at zero. The reference skips the blur
    // entirely once either axis clamps, so an upsample is never filtered.
    const double factor = static_cast<double>(pad_side) / static_cast<double>(target);
    const double sigma = (factor - 1.0) / 2.0;
    return sigma > 0.0 ? static_cast<float>(sigma) : 0.0F;
}

std::vector<float> preprocess_owlv2_image(const float* image_pixels, std::int32_t image_height,
                                          std::int32_t image_width,
                                          const Owlv2PreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("owlv2 source image must be non-empty");
    validate(config);

    // Pad to a square with zeros on the bottom and the right, before any
    // resize. Measured: padding the other way, centring it, or filling with
    // 0.5 each move the pixels far enough to change the detections.
    const std::int32_t side = compute_owlv2_pad_side(image_height, image_width);
    std::vector<float> padded(static_cast<std::size_t>(side) * side * 3, 0.0F);
    for (std::int32_t y = 0; y < image_height; ++y) {
        const auto source = static_cast<std::size_t>(y) * image_width * 3;
        const auto destination = static_cast<std::size_t>(y) * side * 3;
        std::copy_n(image_pixels + source, static_cast<std::size_t>(image_width) * 3,
                    padded.begin() + static_cast<std::ptrdiff_t>(destination));
    }

    const std::int32_t target = config.input_image_h;
    const float sigma = compute_owlv2_antialias_sigma(side, target);
    if (sigma > 0.0F)
        padded = blur(padded, side, sigma);
    const auto resized = resize_bilinear(padded, side, target);

    const auto plane = static_cast<std::size_t>(target) * target;
    std::vector<float> pixel_values(3U * plane);
    for (std::int32_t y = 0; y < target; ++y) {
        for (std::int32_t x = 0; x < target; ++x) {
            const auto source = static_cast<std::size_t>((y * target + x) * 3);
            for (std::int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                pixel_values[channel * plane + static_cast<std::size_t>(y) * target + x] =
                    (resized[source + channel] - config.image_mean[channel]) /
                    config.image_std[channel];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
