/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_nfnet/runtime/image_preprocess_seam.h"

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
    trtmc::TimmNfnetPreprocessConfig config;
    config.input_image_h = 3;
    config.input_image_w = 3;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    const auto values = trtmc::preprocess_timm_nfnet_image(pixels.data(), 2, 2, config);
    check(values.size() == 27, "NFNet preprocess size");
    if (values.size() == 27)
        check_close(values[4], 0.5F, 1e-6F, "NFNet bilinear center pixel");
}

void test_normalization() {
    const std::vector<float> pixels(12U, 0.75F);
    trtmc::TimmNfnetPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.25F, 0.5F, 0.75F};
    config.image_std = {0.5F, 0.25F, 0.125F};
    const auto values = trtmc::preprocess_timm_nfnet_image(pixels.data(), 2, 2, config);
    check_close(values[0], 1.0F, 1e-6F, "NFNet red normalization");
    check_close(values[4], 1.0F, 1e-6F, "NFNet green normalization");
    check_close(values[8], 0.0F, 1e-6F, "NFNet blue normalization");
}

void test_short_edge_geometry() {
    trtmc::TimmNfnetPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.crop_pct = 0.9F;
    const auto landscape = trtmc::compute_timm_nfnet_resize_shape(320, 426, config);
    check(landscape.height == 248, "NFNet landscape short edge");
    check(landscape.width == 330, "NFNet landscape aspect ratio");
    const auto portrait = trtmc::compute_timm_nfnet_resize_shape(426, 320, config);
    check(portrait.height == 330, "NFNet portrait aspect ratio");
    check(portrait.width == 248, "NFNet portrait short edge");
}

void test_invalid_interpolation() {
    bool threw = false;
    try {
        const std::vector<float> pixels(12U, 0.0F);
        trtmc::TimmNfnetPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.crop_pct = 1.0F;
        config.interpolation = "nearest";
        (void)trtmc::preprocess_timm_nfnet_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "NFNet rejects unsupported interpolation");
}

void test_squash_resizes_both_axes_and_centre_keeps_aspect() {
    // dm_nfnet asks for crop_mode "squash": both axes go to floor(size /
    // crop_pct), so a wide image is deliberately distorted. The default
    // "center" keeps the aspect ratio instead. Reading this wrong feeds the
    // engine different pixels while everything still builds and runs.
    trtmc::TimmNfnetPreprocessConfig config;
    config.input_image_h = 192;
    config.input_image_w = 192;
    config.crop_pct = 0.9F;

    config.crop_mode = "squash";
    const auto squashed = trtmc::compute_timm_nfnet_resize_shape(382, 640, config);
    check(squashed.height == 213, "squash resizes the height to floor(size / crop_pct)");
    check(squashed.width == 213, "squash resizes the width to the same length");

    config.crop_mode = "center";
    const auto centred = trtmc::compute_timm_nfnet_resize_shape(382, 640, config);
    check(centred.height == 213, "centre puts the short edge at floor(size / crop_pct)");
    check(centred.width > centred.height, "centre keeps the image wider than it is tall");
}

void test_seam_rejects_an_unknown_crop_mode() {
    bool threw = false;
    try {
        trtmc::TimmNfnetPreprocessConfig config;
        config.crop_mode = "border";
        (void)trtmc::compute_timm_nfnet_resize_shape(382, 640, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "timm NFNet rejects a crop mode it does not implement");
}

} // namespace

int main() {
    test_squash_resizes_both_axes_and_centre_keeps_aspect();
    test_seam_rejects_an_unknown_crop_mode();
    test_bilinear_resize();
    test_normalization();
    test_short_edge_geometry();
    test_invalid_interpolation();
    if (failures)
        std::cerr << failures << " NFNet preprocess test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
