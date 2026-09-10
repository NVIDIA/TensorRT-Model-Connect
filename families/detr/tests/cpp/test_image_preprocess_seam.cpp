/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/detr/runtime/image_preprocess_seam.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::DetrPreprocessConfig plain_config(std::int32_t height, std::int32_t width) {
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = height;
    config.input_image_w = width;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    return config;
}

void test_preprocess_reads_interleaved_and_writes_planar() {
    // The CLI hands over interleaved RGB and the engine wants planar CHW. A
    // constant image cannot tell the two layouts apart, so give each channel
    // its own value.
    constexpr float kRed = 0.10F;
    constexpr float kGreen = 0.50F;
    constexpr float kBlue = 0.90F;
    std::vector<float> pixels(4U * 4U * 3U);
    for (std::size_t pixel = 0; pixel < 16U; ++pixel) {
        pixels[pixel * 3U + 0U] = kRed;
        pixels[pixel * 3U + 1U] = kGreen;
        pixels[pixel * 3U + 2U] = kBlue;
    }
    const auto config = plain_config(4, 4);
    const auto values = trtmc::preprocess_detr_image(pixels.data(), 4, 4, config);

    check(values.size() == 3U * 4U * 4U, "output holds three planes");
    const std::size_t plane = 4U * 4U;
    for (std::size_t index = 0; index < plane; ++index) {
        check_close(values[index], kRed, 1e-6F, "red plane");
        check_close(values[plane + index], kGreen, 1e-6F, "green plane");
        check_close(values[2U * plane + index], kBlue, 1e-6F, "blue plane");
    }
}

void test_preprocess_applies_the_normalisation_per_channel() {
    std::vector<float> pixels(2U * 2U * 3U);
    for (std::size_t pixel = 0; pixel < 4U; ++pixel) {
        pixels[pixel * 3U + 0U] = 0.6F;
        pixels[pixel * 3U + 1U] = 0.6F;
        pixels[pixel * 3U + 2U] = 0.6F;
    }
    trtmc::DetrPreprocessConfig config = plain_config(2, 2);
    config.image_mean = {0.1F, 0.2F, 0.3F};
    config.image_std = {0.5F, 0.25F, 0.2F};
    const auto values = trtmc::preprocess_detr_image(pixels.data(), 2, 2, config);
    const std::size_t plane = 2U * 2U;
    // Each channel uses its own mean and std, never the first one for all.
    check_close(values[0], (0.6F - 0.1F) / 0.5F, 1e-5F, "red normalisation");
    check_close(values[plane], (0.6F - 0.2F) / 0.25F, 1e-5F, "green normalisation");
    check_close(values[2U * plane], (0.6F - 0.3F) / 0.2F, 1e-5F, "blue normalisation");
}

void test_preprocess_resizes_without_preserving_aspect() {
    // DETR normalises its boxes to its own input, so the runtime resizes
    // straight to the engine's shape and maps back with one factor per axis.
    // A gradient across x must survive that resize as a gradient.
    std::vector<float> pixels(2U * 8U * 3U);
    for (std::int32_t y = 0; y < 2; ++y) {
        for (std::int32_t x = 0; x < 8; ++x) {
            const float value = static_cast<float>(x) / 7.0F;
            for (std::int32_t channel = 0; channel < 3; ++channel)
                pixels[(static_cast<std::size_t>(y) * 8U + static_cast<std::size_t>(x)) * 3U +
                       static_cast<std::size_t>(channel)] = value;
        }
    }
    const auto config = plain_config(4, 4);
    const auto values = trtmc::preprocess_detr_image(pixels.data(), 2, 8, config);
    check(values.size() == 3U * 4U * 4U, "resize reaches the configured shape");
    // The first row still runs left to right, low to high.
    check(values[0] < values[3], "resize keeps the horizontal gradient");
    check_close(values[0], 0.0F, 0.2F, "resize keeps the left edge dark");
    check_close(values[3], 1.0F, 0.2F, "resize keeps the right edge bright");
}

void test_preprocess_rejects_an_empty_image() {
    bool threw = false;
    try {
        const auto config = plain_config(4, 4);
        (void)trtmc::preprocess_detr_image(nullptr, 0, 0, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DETR rejects an empty image");
}

void test_preprocess_rejects_a_zero_normalisation() {
    bool threw = false;
    try {
        trtmc::DetrPreprocessConfig config = plain_config(4, 4);
        config.image_std = {1.0F, 0.0F, 1.0F};
        const std::vector<float> pixels(4U * 4U * 3U, 0.5F);
        (void)trtmc::preprocess_detr_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DETR rejects a zero std");
}

} // namespace

int main() {
    test_preprocess_reads_interleaved_and_writes_planar();
    test_preprocess_applies_the_normalisation_per_channel();
    test_preprocess_resizes_without_preserving_aspect();
    test_preprocess_rejects_an_empty_image();
    test_preprocess_rejects_a_zero_normalisation();

    if (g_failures != 0) {
        std::cerr << g_failures << " DETR preprocess test(s) failed\n";
        return 1;
    }
    return 0;
}
