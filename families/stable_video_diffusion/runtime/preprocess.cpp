/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Built with -ffp-contract=off so each float32 operation rounds as in the
// reference's separate tensor operations.

#include "families/stable_video_diffusion/runtime/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trtmc::stable_video_diffusion {

namespace {

constexpr std::size_t kChannels = 3;

std::size_t plane_size(std::int32_t height, std::int32_t width) {
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
}

// _gaussian: exp(-x^2 / (2 sigma^2)) over x = i - window // 2, normalized to sum 1.
std::vector<float> gaussian_kernel(std::int32_t window, float sigma) {
    std::vector<float> kernel(static_cast<std::size_t>(window));
    const float denominator = 2.0F * (sigma * sigma);
    for (std::int32_t index = 0; index < window; ++index) {
        const auto x = static_cast<float>(index - window / 2);
        kernel[static_cast<std::size_t>(index)] = std::exp(-(x * x) / denominator);
    }
    float total = 0.0F;
    for (const float value : kernel)
        total += value;
    for (float& value : kernel)
        value = value / total;
    return kernel;
}

// torch "reflect" padding: mirror about the edge sample without repeating it.
std::int32_t reflect(std::int32_t index, std::int32_t size) {
    if (index < 0)
        return -index;
    if (index >= size)
        return 2 * (size - 1) - index;
    return index;
}

// Depthwise cross-correlation of each CHW plane with a 1-D kernel along one axis.
std::vector<float> blur(const std::vector<float>& planes, std::int32_t height, std::int32_t width,
                        const std::vector<float>& kernel, bool horizontal) {
    const auto window = static_cast<std::int32_t>(kernel.size());
    const std::int32_t before = (window - 1) / 2;
    const std::int32_t extent = horizontal ? width : height;
    if (before >= extent || window - 1 - before >= extent)
        throw std::invalid_argument("Stable Video Diffusion image is too small for its CLIP blur");
    std::vector<float> out(planes.size());
    const std::size_t plane = plane_size(height, width);
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const float* source = planes.data() + channel * plane;
        float* target = out.data() + channel * plane;
        for (std::int32_t y = 0; y < height; ++y) {
            for (std::int32_t x = 0; x < width; ++x) {
                float total = 0.0F;
                for (std::int32_t tap = 0; tap < window; ++tap) {
                    const std::int32_t sy = horizontal ? y : reflect(y + tap - before, height);
                    const std::int32_t sx = horizontal ? reflect(x + tap - before, width) : x;
                    total += kernel[static_cast<std::size_t>(tap)] *
                             source[static_cast<std::size_t>(sy) * width + sx];
                }
                target[static_cast<std::size_t>(y) * width + x] = total;
            }
        }
    }
    return out;
}

struct CubicTaps {
    std::int32_t index[4];
    float weight[4];
};

// ATen's cubic convolution with A = -0.75.
float cubic_inner(float x) {
    constexpr float a = -0.75F;
    return ((a + 2.0F) * x - (a + 3.0F)) * x * x + 1.0F;
}

float cubic_outer(float x) {
    constexpr float a = -0.75F;
    return ((a * x - 5.0F * a) * x + 8.0F * a) * x - 4.0F * a;
}

// Source taps of an align_corners=True bicubic resize; borders are clamped.
std::vector<CubicTaps> cubic_taps(std::int32_t input_size, std::int32_t output_size) {
    const float scale =
        output_size > 1 ? static_cast<float>(input_size - 1) / static_cast<float>(output_size - 1)
                        : 0.0F;
    std::vector<CubicTaps> taps(static_cast<std::size_t>(output_size));
    for (std::int32_t position = 0; position < output_size; ++position) {
        const float real = scale * static_cast<float>(position);
        const std::int32_t base =
            std::min(static_cast<std::int32_t>(std::floor(real)), input_size - 1);
        const float t = std::min(std::max(real - static_cast<float>(base), 0.0F), 1.0F);
        const float mirrored = 1.0F - t;
        auto& entry = taps[static_cast<std::size_t>(position)];
        entry.weight[0] = cubic_outer(t + 1.0F);
        entry.weight[1] = cubic_inner(t);
        entry.weight[2] = cubic_inner(mirrored);
        entry.weight[3] = cubic_outer(mirrored + 1.0F);
        for (std::int32_t tap = 0; tap < 4; ++tap)
            entry.index[tap] = std::clamp(base - 1 + tap, 0, input_size - 1);
    }
    return taps;
}

} // namespace

std::vector<float> initial_image_pixels(const internal::ImageView& image, std::int32_t height,
                                        std::int32_t width) {
    const std::size_t values = plane_size(height, width) * kChannels;
    const bool bytes = image.format == internal::ImageFormat::UInt8;
    if (height <= 0 || width <= 0 || image.data == nullptr || image.channels != kChannels ||
        image.height != static_cast<std::uint32_t>(height) ||
        image.width != static_cast<std::uint32_t>(width) ||
        image.byte_size != values * (bytes ? sizeof(std::uint8_t) : sizeof(float))) {
        throw std::invalid_argument(
            "Stable Video Diffusion needs a contiguous " + std::to_string(width) + "x" +
            std::to_string(height) +
            " RGB initial image, the bundle's video size; resize it to that size first");
    }
    std::vector<float> pixels(values);
    for (std::size_t index = 0; index < values; ++index) {
        const float value =
            bytes ? static_cast<float>(static_cast<const std::uint8_t*>(image.data)[index]) / 255.0F
                  : static_cast<const float*>(image.data)[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument(
                "Stable Video Diffusion initial image values must be in [0, 1]");
        pixels[index] = value;
    }
    return pixels;
}

std::vector<float> vae_image(const std::vector<float>& pixels, std::int32_t height,
                             std::int32_t width) {
    const std::size_t plane = plane_size(height, width);
    if (pixels.size() != plane * kChannels)
        throw std::invalid_argument("Stable Video Diffusion VAE image has the wrong size");
    std::vector<float> chw(pixels.size());
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        for (std::size_t channel = 0; channel < kChannels; ++channel)
            chw[channel * plane + pixel] = 2.0F * pixels[pixel * kChannels + channel] - 1.0F;
    }
    return chw;
}

std::vector<float> clip_pixels(const std::vector<float>& pixels, std::int32_t height,
                               std::int32_t width, std::int32_t size,
                               const std::array<float, 3>& mean, const std::array<float, 3>& std) {
    const std::size_t plane = plane_size(height, width);
    if (size <= 1 || pixels.size() != plane * kChannels)
        throw std::invalid_argument("Stable Video Diffusion CLIP input has the wrong size");
    std::vector<float> planes(pixels.size());
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        for (std::size_t channel = 0; channel < kChannels; ++channel)
            planes[channel * plane + pixel] = pixels[pixel * kChannels + channel] * 2.0F - 1.0F;
    }

    // Python-float arithmetic of _resize_with_antialiasing, then float32 sigmas.
    const double factor_y = static_cast<double>(height) / size;
    const double factor_x = static_cast<double>(width) / size;
    const double sigma_y = std::max((factor_y - 1.0) / 2.0, 0.001);
    const double sigma_x = std::max((factor_x - 1.0) / 2.0, 0.001);
    auto window_y = static_cast<std::int32_t>(std::max(2.0 * 2 * sigma_y, 3.0));
    auto window_x = static_cast<std::int32_t>(std::max(2.0 * 2 * sigma_x, 3.0));
    window_y += window_y % 2 == 0 ? 1 : 0;
    window_x += window_x % 2 == 0 ? 1 : 0;
    planes =
        blur(planes, height, width, gaussian_kernel(window_x, static_cast<float>(sigma_x)), true);
    planes =
        blur(planes, height, width, gaussian_kernel(window_y, static_cast<float>(sigma_y)), false);

    const auto rows = cubic_taps(height, size);
    const auto columns = cubic_taps(width, size);
    const std::size_t output_plane = plane_size(size, size);
    std::vector<float> out(output_plane * kChannels);
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const float* source = planes.data() + channel * plane;
        for (std::int32_t y = 0; y < size; ++y) {
            const auto& row_taps = rows[static_cast<std::size_t>(y)];
            for (std::int32_t x = 0; x < size; ++x) {
                const auto& column_taps = columns[static_cast<std::size_t>(x)];
                float value = 0.0F;
                for (std::int32_t row = 0; row < 4; ++row) {
                    const float* line =
                        source + static_cast<std::size_t>(row_taps.index[row]) * width;
                    float across = line[column_taps.index[0]] * column_taps.weight[0];
                    for (std::int32_t column = 1; column < 4; ++column)
                        across += line[column_taps.index[column]] * column_taps.weight[column];
                    value = row == 0 ? across * row_taps.weight[0]
                                     : value + across * row_taps.weight[row];
                }
                const float unit = (value + 1.0F) / 2.0F;
                out[channel * output_plane + static_cast<std::size_t>(y) * size + x] =
                    (unit - mean[channel]) / std[channel];
            }
        }
    }
    return out;
}

} // namespace trtmc::stable_video_diffusion
