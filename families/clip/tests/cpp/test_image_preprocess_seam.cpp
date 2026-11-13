/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/clip/runtime/image_preprocess_seam.h"

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

trtmc::ClipPreprocessConfig base_config() {
    trtmc::ClipPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    return config;
}

void test_long_edge_is_floored_not_rounded() {
    // 224 * 400 / 300 is 298.67. The reference truncates, so a 300x400 source
    // resizes to 224x298; rounding to 299 shifts every column of the crop by
    // one and still produces a plausible-looking image.
    const auto config = base_config();
    const auto landscape = trtmc::compute_clip_resize_shape(300, 400, config);
    check(landscape.height == 224 && landscape.width == 298, "landscape long edge is floored");

    // The same source turned on its side must transpose, not re-derive.
    const auto portrait = trtmc::compute_clip_resize_shape(400, 300, config);
    check(portrait.height == 298 && portrait.width == 224, "portrait long edge is floored");
}

void test_square_input_is_left_alone() {
    const auto shape = trtmc::compute_clip_resize_shape(224, 224, base_config());
    check(shape.height == 224 && shape.width == 224, "a square source needs no scaling");
}

void test_centre_crop_floors_an_odd_surplus() {
    // 401x300 resizes to 299x224, leaving 75 surplus rows. The reference takes
    // 37 from the top; torchvision's round-half-to-even rule would take 38.
    // The difference is one row of pixels, which no shape check would catch.
    auto config = base_config();
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    const auto shape = trtmc::compute_clip_resize_shape(401, 300, config);
    check(shape.height == 299 && shape.width == 224, "odd surplus resize shape");

    // A horizontal ramp makes the crop offset visible: each source row is a
    // constant, so the first output row names the source row it came from.
    std::vector<float> pixels(static_cast<std::size_t>(401) * 300 * 3);
    for (int y = 0; y < 401; ++y) {
        const float value = static_cast<float>(y) / 400.0F;
        for (int x = 0; x < 300; ++x) {
            const auto index = (static_cast<std::size_t>(y) * 300 + x) * 3;
            pixels[index] = pixels[index + 1] = pixels[index + 2] = value;
        }
    }
    const auto values = trtmc::preprocess_clip_image(pixels.data(), 401, 300, config);
    check(values.size() == 3U * 224U * 224U, "crop output size");
    // Row 37 of a 299-row resize maps back to about 37/299 of the source.
    check_close(values[0], 37.0F / 299.0F, 0.01F, "centre crop starts at the floored offset");
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
    const auto values = trtmc::preprocess_clip_image(pixels.data(), 224, 224, config);
    const std::size_t plane = 224U * 224U;
    check_close(values[0], (0.6F - 0.1F) / 0.5F, 2e-2F, "red channel normalised");
    check_close(values[plane], (0.4F - 0.2F) / 0.25F, 2e-2F, "green channel normalised");
    check_close(values[2 * plane], (0.8F - 0.3F) / 0.125F, 2e-2F, "blue channel normalised");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_clip_image(nullptr, 224, 224, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_clip_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");
}

} // namespace

int main() {
    test_long_edge_is_floored_not_rounded();
    test_square_input_is_left_alone();
    test_centre_crop_floors_an_odd_surplus();
    test_normalisation_is_per_channel();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "clip image preprocess seam: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
