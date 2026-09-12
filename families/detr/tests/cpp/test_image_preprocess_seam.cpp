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

void test_detr_resize_preserves_short_edge_for_square_image() {
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = 800;
    config.input_image_w = 800;

    const auto shape = trtmc::compute_detr_resize_shape(400, 400, config);
    check(shape.height == 800, "detr square resize height");
    check(shape.width == 800, "detr square resize width");
}

void test_detr_resize_caps_longest_edge() {
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = 796;
    config.input_image_w = 1333;

    const auto shape = trtmc::compute_detr_resize_shape(382, 640, config);
    check(shape.height == 796, "detr landscape resize height");
    check(shape.width == 1333, "detr landscape resize width");
}

void test_detr_resize_rounds_half_ties_to_even() {
    const trtmc::DetrPreprocessConfig config;
    const auto landscape = trtmc::compute_detr_resize_shape(600, 1200, config);
    check(landscape.height == 666 && landscape.width == 1333,
          "detr landscape half tie rounds down to even");
    const auto portrait = trtmc::compute_detr_resize_shape(1200, 600, config);
    check(portrait.height == 1333 && portrait.width == 666,
          "detr portrait half tie rounds down to even");
    const auto odd_landscape = trtmc::compute_detr_resize_shape(1335, 2666, config);
    check(odd_landscape.height == 668 && odd_landscape.width == 1333,
          "detr landscape half tie rounds up to even");
    const auto odd_portrait = trtmc::compute_detr_resize_shape(2666, 1335, config);
    check(odd_portrait.height == 1333 && odd_portrait.width == 668,
          "detr portrait half tie rounds up to even");
}

void test_detr_resize_rounds_non_ties_to_nearest() {
    const trtmc::DetrPreprocessConfig config;
    const auto below = trtmc::compute_detr_resize_shape(599, 1200, config);
    check(below.height == 665 && below.width == 1333, "detr resize below half tie");
    const auto above = trtmc::compute_detr_resize_shape(601, 1200, config);
    check(above.height == 668 && above.width == 1333, "detr resize above half tie");
}

void test_detr_preprocess_fits_half_tie_engine_dimensions() {
    const std::vector<float> pixels(3U * 600U * 1200U, 0.75F);
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = 666;
    config.input_image_w = 1333;
    const auto landscape = trtmc::preprocess_detr_image(pixels.data(), 600, 1200, config);
    check(landscape.size() == 3U * 666U * 1333U, "detr landscape fits reference dimensions");

    config.input_image_h = 1333;
    config.input_image_w = 666;
    const auto portrait = trtmc::preprocess_detr_image(pixels.data(), 1200, 600, config);
    check(portrait.size() == 3U * 1333U * 666U, "detr portrait fits reference dimensions");
}

void test_detr_preprocess_applies_normalization() {
    const std::vector<float> pixels(3U * 2U * 2U, 0.75F);
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.shortest_edge = 2;
    config.longest_edge = 2;
    config.image_mean = {0.25F, 0.5F, 0.75F};
    config.image_std = {0.5F, 0.25F, 0.125F};

    const auto pixel_values = trtmc::preprocess_detr_image(pixels.data(), 2, 2, config);
    check(pixel_values.size() == 12, "detr preprocess size");
    check_close(pixel_values[0], 1.0F, 1e-6F, "detr red normalization");
    check_close(pixel_values[4], 1.0F, 1e-6F, "detr green normalization");
    check_close(pixel_values[8], 0.0F, 1e-6F, "detr blue normalization");
}

void test_detr_preprocess_rejects_resize_larger_than_engine_input() {
    bool threw = false;
    try {
        const std::vector<float> pixels(3U * 2U * 2U, 0.0F);
        trtmc::DetrPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.shortest_edge = 4;
        config.longest_edge = 4;
        config.image_mean = {0.0F, 0.0F, 0.0F};
        config.image_std = {1.0F, 1.0F, 1.0F};
        (void)trtmc::preprocess_detr_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "detr rejects a resize larger than the engine input");
}

void test_detr_preprocess_keeps_padding_zero() {
    const std::vector<float> pixels(3U * 2U * 2U, 0.75F);
    trtmc::DetrPreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    config.shortest_edge = 2;
    config.longest_edge = 2;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto pixel_values = trtmc::preprocess_detr_image(pixels.data(), 2, 2, config);
    check_close(pixel_values[0], 0.75F, 1e-6F, "detr valid pixel remains normalized");
    check_close(pixel_values[2], 0.0F, 1e-6F, "detr horizontal padding stays zero");
    check_close(pixel_values[8], 0.0F, 1e-6F, "detr vertical padding stays zero");
}

void test_detr_preprocess_rejects_invalid_config() {
    bool threw = false;
    try {
        const std::vector<float> pixels(12U, 0.0F);
        trtmc::DetrPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.image_std = {0.5F, 0.0F, 0.125F};
        (void)trtmc::preprocess_detr_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "detr rejects zero image std");
}

} // namespace

int main() {
    test_detr_resize_preserves_short_edge_for_square_image();
    test_detr_resize_caps_longest_edge();
    test_detr_resize_rounds_half_ties_to_even();
    test_detr_resize_rounds_non_ties_to_nearest();
    test_detr_preprocess_fits_half_tie_engine_dimensions();
    test_detr_preprocess_applies_normalization();
    test_detr_preprocess_rejects_invalid_config();
    test_detr_preprocess_rejects_resize_larger_than_engine_input();
    test_detr_preprocess_keeps_padding_zero();

    if (g_failures != 0) {
        std::cerr << g_failures << " detr preprocess test(s) failed\n";
        return 1;
    }
    return 0;
}
