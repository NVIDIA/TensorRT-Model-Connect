/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

trtmc::Dinov2PreprocessConfig checkpoint_config() {
    return {};
}

void test_transformers_resize_and_crop_geometry() {
    const auto config = checkpoint_config();
    // Landscape 640x480 (WxH): short edge 256, long edge int(256 * 640 / 480) = 341.
    auto geometry = trtmc::compute_dinov2_image_geometry(480, 640, config);
    require(geometry.resized_h == 256 && geometry.resized_w == 341,
            "landscape resize must floor the long edge");
    require(geometry.crop_y == 16 && geometry.crop_x == 58, "center crop must floor its offset");
    // Portrait 333x500 (WxH): long edge int(256 * 500 / 333) = 384, crop offsets floor.
    geometry = trtmc::compute_dinov2_image_geometry(500, 333, config);
    require(geometry.resized_h == 384 && geometry.resized_w == 256,
            "portrait resize must fix the width");
    require(geometry.crop_y == 80 && geometry.crop_x == 16, "portrait crop offsets");
    // Odd margins: (257 - 224) / 2 floors to 16, unlike torchvision's round-half-even.
    geometry =
        trtmc::compute_dinov2_image_geometry(257, 256, {224, 224, 256, {0, 0, 0}, {1, 1, 1}});
    require(geometry.resized_h == 257 && geometry.crop_y == 16, "odd margins floor");
}

void test_constant_image_normalization() {
    const auto config = checkpoint_config();
    const std::vector<uint8_t> image(300U * 400U * 3U, 128);
    const auto pixels = trtmc::preprocess_dinov2_image(image.data(), 300, 400, config);
    require(pixels.size() == 3U * 224U * 224U, "output is one NCHW crop");
    const auto plane = pixels.size() / 3;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float expected = (static_cast<float>(128) / 255.0F - config.image_mean[channel]) /
                               config.image_std[channel];
        for (std::size_t index = 0; index < plane; ++index) {
            if (pixels[channel * plane + index] != expected)
                throw std::runtime_error("normalized bicubic resize must preserve a flat image");
        }
    }
}

void test_identity_resize_preserves_pixels() {
    // A source already at the resize size takes neither Pillow pass.
    const trtmc::Dinov2PreprocessConfig config{2, 2, 2, {0, 0, 0}, {1, 1, 1}};
    const std::vector<uint8_t> image{0, 51, 102, 153, 204, 255, 10, 20, 30, 40, 50, 60};
    const auto pixels = trtmc::preprocess_dinov2_image(image.data(), 2, 2, config);
    require(pixels[0] == 0.0F && pixels[4] == static_cast<float>(51) / 255.0F &&
                pixels[11] == static_cast<float>(60) / 255.0F,
            "identity resize must keep 8-bit values and NCHW order");
}

// Straightforward reference: Pillow's two-pass resize of the whole image (64-bit sums),
// then the Transformers center crop, rescale and normalize.
// Pillow's Resample.c bicubic_filter with a = -0.5, in its evaluation order.
double reference_cubic(double value) {
    value = std::abs(value);
    if (value < 1.0)
        return ((-0.5 + 2.0) * value - (-0.5 + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return (((value - 5.0) * value + 8.0) * value - 4.0) * -0.5;
    return 0.0;
}

std::vector<uint8_t> reference_resize(const std::vector<uint8_t>& input, int32_t in_h, int32_t in_w,
                                      int32_t out_h, int32_t out_w) {
    auto pass = [](const std::vector<uint8_t>& source, int32_t rows, int32_t in_size,
                   int32_t out_size, bool horizontal, int32_t other) {
        const double scale = static_cast<double>(in_size) / out_size;
        const double filter_scale = std::max(scale, 1.0);
        const double support = 2.0 * filter_scale;
        std::vector<uint8_t> result(static_cast<std::size_t>(rows) * out_size * 3U);
        for (int32_t o = 0; o < out_size; ++o) {
            const double center = (o + 0.5) * scale;
            const int32_t first = std::max(0, static_cast<int32_t>(center - support + 0.5));
            const int32_t end = std::min(in_size, static_cast<int32_t>(center + support + 0.5));
            std::vector<double> weights;
            double total = 0.0;
            for (int32_t i = first; i < end; ++i) {
                weights.push_back(reference_cubic((i - center + 0.5) * (1.0 / filter_scale)));
                total += weights.back();
            }
            std::vector<int64_t> fixed;
            for (double w : weights) {
                const double scaled = w / total * 4194304.0;
                fixed.push_back(static_cast<int64_t>(scaled < 0 ? scaled - 0.5 : scaled + 0.5));
            }
            for (int32_t r = 0; r < rows; ++r) {
                for (int32_t c = 0; c < 3; ++c) {
                    int64_t sum = 2097152;
                    for (int32_t i = first; i < end; ++i) {
                        const std::size_t at =
                            horizontal ? (static_cast<std::size_t>(r) * in_size + i) * 3U + c
                                       : (static_cast<std::size_t>(i) * other + r) * 3U + c;
                        sum += source[at] * fixed[static_cast<std::size_t>(i - first)];
                    }
                    const uint8_t value =
                        sum <= 0 ? 0 : static_cast<uint8_t>(std::min<int64_t>(sum >> 22, 255));
                    const std::size_t at =
                        horizontal ? (static_cast<std::size_t>(r) * out_size + o) * 3U + c
                                   : (static_cast<std::size_t>(o) * other + r) * 3U + c;
                    result[at] = value;
                }
            }
        }
        return result;
    };
    auto horizontal = in_w == out_w ? input : pass(input, in_h, in_w, out_w, true, 0);
    return in_h == out_h ? horizontal : pass(horizontal, out_w, in_h, out_h, false, out_w);
}

std::vector<float> reference_preprocess(const std::vector<uint8_t>& image, int32_t h, int32_t w,
                                        const trtmc::Dinov2PreprocessConfig& config) {
    const auto geometry = trtmc::compute_dinov2_image_geometry(h, w, config);
    const auto resized = reference_resize(image, h, w, geometry.resized_h, geometry.resized_w);
    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> result(3U * plane);
    for (int32_t y = 0; y < config.input_image_h; ++y) {
        for (int32_t x = 0; x < config.input_image_w; ++x) {
            const auto source =
                (static_cast<std::size_t>(geometry.crop_y + y) * geometry.resized_w +
                 static_cast<std::size_t>(geometry.crop_x + x)) *
                3U;
            for (std::size_t c = 0; c < 3; ++c) {
                const float value = static_cast<float>(resized[source + c]) / 255.0F;
                result[c * plane + static_cast<std::size_t>(y) * config.input_image_w + x] =
                    (value - config.image_mean[c]) / config.image_std[c];
            }
        }
    }
    return result;
}

void test_crop_window_matches_full_resize_then_crop() {
    struct Case {
        int32_t height, width;
        trtmc::Dinov2PreprocessConfig config;
    };
    const trtmc::Dinov2PreprocessConfig checkpoint{};
    const trtmc::Dinov2PreprocessConfig small{14, 28, 32, {0.5F, 0.25F, 0.75F}, {0.2F, 0.4F, 0.8F}};
    const std::vector<Case> cases{
        {382, 640, checkpoint},   // the E2E photograph: both passes downscale
        {640, 382, checkpoint},   // portrait
        {1500, 2000, checkpoint}, // large downscale
        {256, 256, checkpoint},   // no resize at all
        {256, 300, checkpoint},   // horizontal pass only
        {300, 256, checkpoint},   // vertical pass only
        {257, 256, checkpoint},   // odd crop margin
        {150, 100, checkpoint},   // upscale in both directions
        {7, 10, small},           // tiny upscale with a narrow non-square crop
        {45, 33, small},
    };
    uint32_t state = 12345;
    for (const auto& item : cases) {
        std::vector<uint8_t> image(static_cast<std::size_t>(item.height) * item.width * 3U);
        for (auto& value : image) {
            state = state * 1664525U + 1013904223U;
            value = static_cast<uint8_t>(state >> 24);
        }
        const auto actual =
            trtmc::preprocess_dinov2_image(image.data(), item.height, item.width, item.config);
        const auto expected = reference_preprocess(image, item.height, item.width, item.config);
        if (actual != expected)
            throw std::runtime_error("crop-window resampling must be bit-identical to "
                                     "resizing the whole image and cropping it");
    }
}

void test_invalid_configuration() {
    const std::vector<uint8_t> image(12, 0);
    rejects([&] { trtmc::compute_dinov2_image_geometry(0, 2, checkpoint_config()); },
            "reject empty source");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {4, 4, 2, {0, 0, 0}, {1, 1, 1}}); },
            "reject crop larger than resize");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {2, 2, 2, {0, 0}, {1, 1, 1}}); },
            "reject short mean");
    rejects([&] { trtmc::compute_dinov2_image_geometry(2, 2, {2, 2, 2, {0, 0, 0}, {1, 0, 1}}); },
            "reject zero std");
    rejects([&] { trtmc::preprocess_dinov2_image(nullptr, 2, 2, checkpoint_config()); },
            "reject null pixels");
}

} // namespace

int main() {
    try {
        test_transformers_resize_and_crop_geometry();
        test_constant_image_normalization();
        test_identity_resize_preserves_pixels();
        test_crop_window_matches_full_resize_then_crop();
        test_invalid_configuration();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    std::cout << "dinov2 image preprocessing tests passed\n";
    return 0;
}
