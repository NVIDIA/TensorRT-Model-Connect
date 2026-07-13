/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-03-SEGFORMER
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SegFormer-owned preprocessing seam: image normalization
// Preconditions:  Decoded image data is available
// Postconditions: Normalization produces correct float values, empty images rejected
// =============================================================================

#include "runtime/models/segformer/segformer_preprocess_seam.h"

#include <cmath>
#include <cstdint>
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

trtmc::runtime::adapters::io::DecodedImage make_two_pixel_image() {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = 2;
    image.height = 1;
    image.channels = 3;
    image.pixels = {
        255, 0, 0, 0, 255, 0,
    };
    return image;
}

void test_segformer_preprocess_normalizes_decoded_image() {
    trtmc::SegformerPreprocessConfig config;
    config.input_image_h = 1;
    config.input_image_w = 2;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto pixel_values = trtmc::preprocess_segformer_image(make_two_pixel_image(), config);
    check(pixel_values.size() == 6, "segformer preprocess size");
    if (pixel_values.size() != 6) {
        return;
    }

    check_close(pixel_values[0], 1.0F, 1e-6F, "segformer preprocess red channel pixel 0");
    check_close(pixel_values[1], 0.0F, 1e-6F, "segformer preprocess red channel pixel 1");
    check_close(pixel_values[2], 0.0F, 1e-6F, "segformer preprocess green channel pixel 0");
    check_close(pixel_values[3], 1.0F, 1e-6F, "segformer preprocess green channel pixel 1");
}

void test_segformer_preprocess_rejects_empty_image() {
    bool threw = false;
    try {
        trtmc::SegformerPreprocessConfig config;
        (void)trtmc::preprocess_segformer_image({}, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "segformer preprocess rejects empty image");
}

void test_segformer_preprocess_uses_bilinear_resize() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    trtmc::SegformerPreprocessConfig config;
    config.input_image_h = 3;
    config.input_image_w = 3;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto pixel_values = trtmc::preprocess_segformer_image(pixels.data(), 2, 2, config);
    check(pixel_values.size() == 27, "segformer bilinear resize size");
    if (pixel_values.size() == 27) {
        check_close(pixel_values[4], 0.5F, 1e-6F, "segformer bilinear resize blends center pixel");
    }
}

} // namespace

int main() {
    test_segformer_preprocess_normalizes_decoded_image();
    test_segformer_preprocess_rejects_empty_image();
    test_segformer_preprocess_uses_bilinear_resize();

    if (g_failures != 0) {
        std::cerr << g_failures << " SegFormer preprocess seam test(s) failed\n";
        return 1;
    }
    return 0;
}
