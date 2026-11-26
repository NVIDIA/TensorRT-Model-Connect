/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace trtmc {
namespace {

void validate(const DepthAnythingPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0)
        throw std::invalid_argument("depth_anything engine input dimensions must be positive");
    if (config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::invalid_argument("depth_anything image mean/std must contain three channels");
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("depth_anything image std must be non-zero");
    }
}

// Pillow's bilinear resample of 8-bit images, which the reference image processor uses: an
// antialiased triangle filter with 22-bit fixed-point taps, a horizontal pass to an 8-bit image,
// then a vertical pass.
constexpr int kPrecisionBits = 32 - 8 - 2;

struct Taps {
    std::vector<int32_t> first;
    std::vector<int32_t> count;
    std::vector<int32_t> weights; // `width` fixed-point taps per output position
    int32_t width{0};
};

Taps bilinear_taps(int32_t in_size, int32_t out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = filter_scale; // the triangle filter's support is 1
    Taps taps;
    taps.width = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    taps.first.resize(static_cast<std::size_t>(out_size));
    taps.count.resize(static_cast<std::size_t>(out_size));
    taps.weights.assign(static_cast<std::size_t>(out_size) * taps.width, 0);
    std::vector<double> weights(static_cast<std::size_t>(taps.width));
    for (int32_t out = 0; out < out_size; ++out) {
        const double center = (out + 0.5) * scale;
        const auto first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const auto last = std::min(static_cast<int32_t>(center + support + 0.5), in_size);
        const int32_t count = last - first;
        double total = 0.0;
        for (int32_t k = 0; k < count; ++k) {
            const double distance = std::fabs((k + first - center + 0.5) / filter_scale);
            weights[static_cast<std::size_t>(k)] = distance < 1.0 ? 1.0 - distance : 0.0;
            total += weights[static_cast<std::size_t>(k)];
        }
        for (int32_t k = 0; k < count; ++k) {
            const double weight = total != 0.0 ? weights[static_cast<std::size_t>(k)] / total : 0.0;
            const double fixed = weight * static_cast<double>(1 << kPrecisionBits);
            taps.weights[static_cast<std::size_t>(out) * taps.width + k] =
                static_cast<int32_t>(fixed < 0.0 ? fixed - 0.5 : fixed + 0.5);
        }
        taps.first[static_cast<std::size_t>(out)] = first;
        taps.count[static_cast<std::size_t>(out)] = count;
    }
    return taps;
}

uint8_t clip8(int32_t sum) {
    return static_cast<uint8_t>(std::clamp(sum >> kPrecisionBits, 0, 255));
}

// Horizontal pass of an interleaved RGB image: [rows][in_w][3] -> [rows][out_w][3].
std::vector<uint8_t> resample_horizontal(const std::vector<uint8_t>& source, int32_t rows,
                                         int32_t in_w, int32_t out_w) {
    const Taps taps = bilinear_taps(in_w, out_w);
    std::vector<uint8_t> output(static_cast<std::size_t>(rows) * out_w * 3U);
    for (int32_t row = 0; row < rows; ++row) {
        const uint8_t* line = &source[static_cast<std::size_t>(row) * in_w * 3U];
        uint8_t* target = &output[static_cast<std::size_t>(row) * out_w * 3U];
        for (int32_t out = 0; out < out_w; ++out) {
            const int32_t* weights = &taps.weights[static_cast<std::size_t>(out) * taps.width];
            const uint8_t* first = line + static_cast<std::size_t>(taps.first[out]) * 3U;
            int32_t r = 1 << (kPrecisionBits - 1), g = r, b = r;
            const int32_t count = taps.count[out];
            for (int32_t k = 0; k < count; ++k) {
                r += first[3 * k] * weights[k];
                g += first[3 * k + 1] * weights[k];
                b += first[3 * k + 2] * weights[k];
            }
            target[3 * out] = clip8(r);
            target[3 * out + 1] = clip8(g);
            target[3 * out + 2] = clip8(b);
        }
    }
    return output;
}

// Vertical pass: [in_h][width][3] -> [out_h][width][3], accumulating whole contiguous rows.
std::vector<uint8_t> resample_vertical(const std::vector<uint8_t>& source, int32_t width,
                                       int32_t in_h, int32_t out_h) {
    const Taps taps = bilinear_taps(in_h, out_h);
    const std::size_t row_values = static_cast<std::size_t>(width) * 3U;
    std::vector<uint8_t> output(static_cast<std::size_t>(out_h) * row_values);
    std::vector<int32_t> sums(row_values);
    for (int32_t out = 0; out < out_h; ++out) {
        std::fill(sums.begin(), sums.end(), 1 << (kPrecisionBits - 1));
        const int32_t* weights = &taps.weights[static_cast<std::size_t>(out) * taps.width];
        for (int32_t k = 0; k < taps.count[out]; ++k) {
            const uint8_t* row =
                &source[static_cast<std::size_t>(taps.first[out] + k) * row_values];
            for (std::size_t index = 0; index < row_values; ++index)
                sums[index] += row[index] * weights[k];
        }
        uint8_t* target = &output[static_cast<std::size_t>(out) * row_values];
        for (std::size_t index = 0; index < row_values; ++index)
            target[index] = clip8(sums[index]);
    }
    return output;
}

} // namespace

std::vector<float> preprocess_depth_anything_image(const float* image_pixels, int32_t image_height,
                                                   int32_t image_width,
                                                   const DepthAnythingPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0)
        throw std::invalid_argument("depth_anything source image must be non-empty");
    validate(config);

    const int32_t out_h = config.input_image_h;
    const int32_t out_w = config.input_image_w;
    // The caller's pixels are 8-bit values scaled to [0, 1]; the reference resamples the 8-bit
    // image.
    std::vector<uint8_t> image(static_cast<std::size_t>(image_height) * image_width * 3U);
    for (std::size_t index = 0; index < image.size(); ++index)
        image[index] =
            static_cast<uint8_t>(static_cast<int32_t>(image_pixels[index] * 255.0F + 0.5F));
    if (image_width != out_w)
        image = resample_horizontal(image, image_height, image_width, out_w);
    if (image_height != out_h)
        image = resample_vertical(image, out_w, image_height, out_h);
    // Every 8-bit value of a channel normalizes to one float: (value / 255 - mean) / std.
    float normalized[3][256];
    for (int channel = 0; channel < 3; ++channel)
        for (int value = 0; value < 256; ++value)
            normalized[channel][value] =
                (static_cast<float>(value) / 255.0F - config.image_mean[channel]) /
                config.image_std[channel];

    const auto plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::size_t index = 0; index < plane; ++index) {
        const uint8_t* pixel = &image[index * 3U];
        pixel_values[index] = normalized[0][pixel[0]];
        pixel_values[plane + index] = normalized[1][pixel[1]];
        pixel_values[2U * plane + index] = normalized[2][pixel[2]];
    }
    return pixel_values;
}

} // namespace trtmc
