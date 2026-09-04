/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_inception/runtime/image_preprocess_seam.h"

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

void test_timm_inception_preprocess_uses_configured_bilinear_resize() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    trtmc::TimmInceptionPreprocessConfig config;
    config.input_image_h = 3;
    config.input_image_w = 3;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto pixel_values = trtmc::preprocess_timm_inception_image(pixels.data(), 2, 2, config);
    check(pixel_values.size() == 27, "timm Inception preprocess size");
    if (pixel_values.size() == 27) {
        check_close(pixel_values[4], 0.5F, 1e-6F,
                    "timm Inception bilinear resize blends center pixel");
    }
}

void test_timm_inception_preprocess_applies_bundle_normalization() {
    const std::vector<float> pixels(3U * 2U * 2U, 0.75F);
    trtmc::TimmInceptionPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.25F, 0.5F, 0.75F};
    config.image_std = {0.5F, 0.25F, 0.125F};

    const auto pixel_values = trtmc::preprocess_timm_inception_image(pixels.data(), 2, 2, config);
    check_close(pixel_values[0], 1.0F, 1e-6F, "timm Inception red normalization");
    check_close(pixel_values[4], 1.0F, 1e-6F, "timm Inception green normalization");
    check_close(pixel_values[8], 0.0F, 1e-6F, "timm Inception blue normalization");
}

void test_timm_inception_resize_matches_torchvision_short_edge_geometry() {
    trtmc::TimmInceptionPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.crop_pct = 0.9F;
    config.interpolation = "bicubic";
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto landscape = trtmc::compute_timm_inception_resize_shape(320, 426, config);
    check(landscape.height == 248, "timm Inception landscape short edge uses floor size");
    check(landscape.width == 330, "timm Inception landscape aspect ratio uses floor size");

    const auto portrait = trtmc::compute_timm_inception_resize_shape(426, 320, config);
    check(portrait.height == 330, "timm Inception portrait aspect ratio uses floor size");
    check(portrait.width == 248, "timm Inception portrait short edge uses floor size");
}

void test_timm_inception_preprocess_rejects_invalid_interpolation() {
    bool threw = false;
    try {
        const std::vector<float> pixels(12U, 0.0F);
        trtmc::TimmInceptionPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.crop_pct = 1.0F;
        config.interpolation = "nearest";
        config.image_mean = {0.0F, 0.0F, 0.0F};
        config.image_std = {1.0F, 1.0F, 1.0F};
        (void)trtmc::preprocess_timm_inception_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "timm Inception rejects unsupported interpolation");
}

} // namespace

int main() {
    test_timm_inception_preprocess_uses_configured_bilinear_resize();
    test_timm_inception_preprocess_applies_bundle_normalization();
    test_timm_inception_resize_matches_torchvision_short_edge_geometry();
    test_timm_inception_preprocess_rejects_invalid_interpolation();

    if (g_failures != 0) {
        std::cerr << g_failures << " timm Inception preprocess test(s) failed\n";
        return 1;
    }
    return 0;
}
