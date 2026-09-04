/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_repvgg/runtime/image_preprocess_seam.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void test_bilinear_resize() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    trtmc::TimmRepvggPreprocessConfig config;
    config.input_image_h = 3;
    config.input_image_w = 3;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    const auto values = trtmc::preprocess_timm_repvgg_image(pixels.data(), 2, 2, config);
    check(values.size() == 27, "RepVGG preprocess size");
    if (values.size() == 27)
        check_close(values[4], 0.5F, 1e-6F, "RepVGG bilinear center pixel");
}

void test_normalization() {
    const std::vector<float> pixels(12U, 0.75F);
    trtmc::TimmRepvggPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.25F, 0.5F, 0.75F};
    config.image_std = {0.5F, 0.25F, 0.125F};
    const auto values = trtmc::preprocess_timm_repvgg_image(pixels.data(), 2, 2, config);
    check_close(values[0], 1.0F, 1e-6F, "RepVGG red normalization");
    check_close(values[4], 1.0F, 1e-6F, "RepVGG green normalization");
    check_close(values[8], 0.0F, 1e-6F, "RepVGG blue normalization");
}

void test_short_edge_geometry() {
    trtmc::TimmRepvggPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.crop_pct = 0.9F;
    const auto landscape = trtmc::compute_timm_repvgg_resize_shape(320, 426, config);
    check(landscape.height == 248, "RepVGG landscape short edge");
    check(landscape.width == 330, "RepVGG landscape aspect ratio");
    const auto portrait = trtmc::compute_timm_repvgg_resize_shape(426, 320, config);
    check(portrait.height == 330, "RepVGG portrait aspect ratio");
    check(portrait.width == 248, "RepVGG portrait short edge");
}

void test_invalid_interpolation() {
    bool threw = false;
    try {
        const std::vector<float> pixels(12U, 0.0F);
        trtmc::TimmRepvggPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.crop_pct = 1.0F;
        config.interpolation = "nearest";
        (void)trtmc::preprocess_timm_repvgg_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "RepVGG rejects unsupported interpolation");
}

} // namespace

int main() {
    test_bilinear_resize();
    test_normalization();
    test_short_edge_geometry();
    test_invalid_interpolation();
    if (failures)
        std::cerr << failures << " RepVGG preprocess test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
