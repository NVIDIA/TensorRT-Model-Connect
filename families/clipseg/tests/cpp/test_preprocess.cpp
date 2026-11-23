/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/clipseg/runtime/image_preprocess_seam.h"

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
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        !(std::fabs(actual - expected) <= tolerance)) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::ClipsegPreprocessConfig base_config() {
    trtmc::ClipsegPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    return config;
}

void test_resize_preserves_the_full_rectangular_source() {
    const auto config = base_config();
    for (const auto shape : {trtmc::compute_clipseg_resize_shape(300, 400, config),
                             trtmc::compute_clipseg_resize_shape(400, 300, config),
                             trtmc::compute_clipseg_resize_shape(224, 224, config)}) {
        check(shape.height == 224 && shape.width == 224, "direct resize uses both target axes");
    }
}

void test_resize_matches_pillow_bilinear_pixels() {
    auto config = base_config();
    config.input_image_h = config.input_image_w = 3;
    config.image_mean = {0, 0, 0};
    config.image_std = {1, 1, 1};
    std::vector<float> source(24);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index * 10) / 255.0F;
    // Pillow RGB resize of the 2x4 source to 3x3, BILINEAR; planar channel order.
    const std::vector<int> expected = {9,  45, 81,  69, 105, 141, 129, 165, 201,
                                       19, 55, 91,  79, 115, 151, 139, 175, 211,
                                       29, 65, 101, 89, 125, 161, 149, 185, 221};
    const auto result = trtmc::preprocess_clipseg_image(source.data(), 2, 4, config);
    check(result.size() == expected.size(), "Pillow golden output shape");
    for (std::size_t index = 0; index < expected.size(); ++index)
        check_close(result[index], static_cast<float>(expected[index]) / 255.0F, 1e-7F,
                    "direct bilinear resize matches Pillow");
}

void test_normalisation_is_per_channel() {
    // A flat image cannot tell interleaved from planar, so each channel gets
    // its own value and its own statistics.
    auto config = base_config();
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.image_mean = {0.1F, 0.2F, 0.3F};
    config.image_std = {0.5F, 0.25F, 0.125F};
    std::vector<float> pixels(static_cast<std::size_t>(224) * 224 * 3);
    for (std::size_t pixel = 0; pixel < 224U * 224U; ++pixel) {
        pixels[pixel * 3] = 0.6F;
        pixels[pixel * 3 + 1] = 0.4F;
        pixels[pixel * 3 + 2] = 0.8F;
    }
    const auto values = trtmc::preprocess_clipseg_image(pixels.data(), 224, 224, config);
    const std::size_t plane = 224U * 224U;
    check_close(values[0], (0.6F - 0.1F) / 0.5F, 2e-2F, "red channel normalised");
    check_close(values[plane], (0.4F - 0.2F) / 0.25F, 2e-2F, "green channel normalised");
    check_close(values[2 * plane], (0.8F - 0.3F) / 0.125F, 2e-2F, "blue channel normalised");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_clipseg_image(nullptr, 224, 224, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_clipseg_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");
}

} // namespace

int main() {
    test_resize_preserves_the_full_rectangular_source();
    test_resize_matches_pillow_bilinear_pixels();
    test_normalisation_is_per_channel();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "clipseg image preprocess seam: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
