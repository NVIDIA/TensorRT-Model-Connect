/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/vision_pixels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::muse_glimmer {
namespace {
std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Muse visual buffer dimensions overflow");
    return a * b;
}
float cubic(float x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5F * x - 2.5F) * x) * x + 1.0F;
    if (x < 2)
        return ((-0.5F * x + 2.5F) * x - 4.0F) * x + 2.0F;
    return 0;
}
using Taps = std::vector<std::vector<std::pair<std::size_t, float>>>;
Taps resize_axis(std::size_t input, int output) {
    const double step = static_cast<double>(input) / output, radius = std::max(1.0, step);
    Taps taps(output);
    for (int o = 0; o < output; ++o) {
        const double center = (o + 0.5) * step - 0.5;
        const auto lo = static_cast<std::int64_t>(std::ceil(center - 2 * radius));
        const auto hi = static_cast<std::int64_t>(std::floor(center + 2 * radius));
        float total = 0;
        for (auto i = lo; i <= hi; ++i) {
            const float weight = cubic(static_cast<float>((i - center) / radius));
            if (weight == 0)
                continue;
            const auto j = static_cast<std::size_t>(std::clamp<std::int64_t>(i, 0, input - 1));
            taps[o].emplace_back(j, weight);
            total += weight;
        }
        for (auto& tap : taps[o])
            tap.second /= total;
    }
    return taps;
}
std::vector<float> resize_rgb(const internal::ImageView& image, int height, int width) {
    const auto rgb = visual_rgb(image);
    const auto x_weights = resize_axis(image.width, width);
    const auto y_weights = resize_axis(image.height, height);
    const auto stride = checked_product(width, 3);
    std::vector<float> horizontal(checked_product(image.height, stride), 0.0F);
    for (std::size_t y = 0; y < image.height; ++y)
        for (int x = 0; x < width; ++x)
            for (auto [i, weight] : x_weights[x])
                for (int c = 0; c < 3; ++c)
                    horizontal[y * stride + 3 * x + c] +=
                        weight * rgb[(y * image.width + i) * 3 + c];
    std::vector<float> result(checked_product(height, stride), 0.0F);
    for (int y = 0; y < height; ++y)
        for (auto [i, weight] : y_weights[y])
            for (std::size_t x = 0; x < stride; ++x)
                result[y * stride + x] += weight * horizontal[i * stride + x];
    for (auto& pixel : result) {
        const float rgb = std::floor(std::clamp(pixel, 0.0F, 255.0F) + 0.5F);
        pixel = rgb * (2.0F / 255.0F) - 1.0F;
    }
    return result;
}
} // namespace
std::vector<std::uint8_t> visual_rgb(const internal::ImageView& image) {
    const bool bytes = image.format == internal::ImageFormat::UInt8;
    if ((!bytes && image.format != internal::ImageFormat::Float32) || !image.data || !image.width ||
        !image.height || (image.channels != 1 && image.channels != 3 && image.channels != 4))
        throw std::invalid_argument("Muse expects a nonempty grayscale, RGB, or RGBA image");
    const auto count = checked_product(checked_product(image.width, image.height), image.channels);
    if (image.byte_size != checked_product(count, bytes ? 1 : sizeof(float)))
        throw std::invalid_argument("Muse image dimensions and byte size disagree");
    const auto* raw = static_cast<const std::uint8_t*>(image.data);
    auto sample = [&](std::size_t pixel, int channel) {
        auto i = pixel * image.channels + (image.channels == 1 ? 0 : channel);
        if (bytes)
            return static_cast<float>(raw[i]);
        float v;
        std::memcpy(&v, raw + i * sizeof(float), sizeof(float));
        if (!std::isfinite(v) || v < 0 || v > 1)
            throw std::invalid_argument("Muse float image expects finite values in [0,1]");
        return std::floor(v * 255.0F + 0.5F);
    };
    std::vector<std::uint8_t> output(checked_product(count / image.channels, 3));
    for (std::size_t i = 0; i < output.size(); ++i)
        output[i] = static_cast<std::uint8_t>(sample(i / 3, i % 3));
    return output;
}
std::vector<float> visual_patches(const internal::ImageView& first,
                                  const internal::ImageView* second, int rows, int cols,
                                  int patch_size) {
    if (rows < 2 || cols < 2 || rows % 2 || cols % 2 || patch_size <= 0 ||
        std::max(rows, cols) > std::numeric_limits<int>::max() / patch_size)
        throw std::invalid_argument("Invalid Muse temporal patch geometry");
    if (second && (second->width != first.width || second->height != first.height))
        throw std::invalid_argument("Muse video frames have inconsistent geometry");
    const int height = rows * patch_size, width = cols * patch_size;
    auto first_pixels = resize_rgb(first, height, width);
    auto next = second ? resize_rgb(*second, height, width) : std::vector<float>{};
    const auto& second_pixels = second ? next : first_pixels;
    std::vector<float> packed;
    packed.reserve(checked_product(2, first_pixels.size()));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            for (int t = 0; t < 2; ++t)
                for (int c = 0; c < 3; ++c)
                    for (int dy = 0; dy < patch_size; ++dy)
                        for (int dx = 0; dx < patch_size; ++dx) {
                            const auto i = (static_cast<std::size_t>(y * patch_size + dy) * width +
                                            x * patch_size + dx) *
                                               3 +
                                           c;
                            packed.push_back((t ? second_pixels : first_pixels)[i]);
                        }
    return packed;
}
} // namespace trtmc::muse_glimmer
