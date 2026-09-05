/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov3/runtime/image_preprocess.h"

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

trtmc::Dinov3PreprocessConfig identity_config(int32_t height, int32_t width) {
    trtmc::Dinov3PreprocessConfig config;
    config.input_image_h = height;
    config.input_image_w = width;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    return config;
}

void test_defaults_match_hugging_face_processor() {
    const trtmc::Dinov3PreprocessConfig config;
    check(config.input_image_h == 224 && config.input_image_w == 224,
          "DINOv3 default processor size");
    check(config.image_mean == std::vector<float>({0.485F, 0.456F, 0.406F}),
          "DINOv3 ImageNet mean");
    check(config.image_std == std::vector<float>({0.229F, 0.224F, 0.225F}), "DINOv3 ImageNet std");
}

void test_direct_resize_preserves_rectangular_image_extent() {
    // A direct 1x2 -> 2x2 resize retains both columns. A shortest-edge resize
    // plus center crop would discard part of this red/blue extent.
    const std::vector<float> pixels = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 1, 2, identity_config(2, 2));
    const std::vector<float> expected = {
        1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
    };
    check(output == expected, "DINOv3 direct resize and NCHW layout");
}

void test_bilinear_resize_blends_center_pixel() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 2, 2, identity_config(3, 3));
    check(output.size() == 27, "DINOv3 bilinear output size");
    if (output.size() == 27)
        check_close(output[4], 0.5F, 1e-6F, "DINOv3 bilinear center value");
}

void test_resize_matches_hugging_face_torchvision_golden() {
    // Golden from DINOv3ViTImageProcessor's torchvision backend:
    // rescale to float, direct bilinear resize to 3x3 with antialiasing, NCHW.
    const std::vector<float> pixels_u8 = {
        0,   10, 20,  64,  74, 84, 192, 202, 212, 255, 245, 235,
        255, 0,  128, 192, 32, 96, 64,  224, 48,  0,   255, 16,
    };
    std::vector<float> pixels = pixels_u8;
    for (float& value : pixels)
        value /= 255.0F;
    const std::vector<float> expected = {
        0.0752941221F, 0.5019608140F, 0.9258823991F, 0.5005882382F, 0.5019608140F, 0.5005882382F,
        0.9258823395F, 0.5019608140F, 0.0752940923F, 0.1145098135F, 0.5411764979F, 0.9101961255F,
        0.0760784373F, 0.5215686560F, 0.9368628263F, 0.0376470610F, 0.5019608140F, 0.9635294676F,
        0.1537255049F, 0.5803921819F, 0.8945097923F, 0.3090196252F, 0.4313725829F, 0.4974509776F,
        0.4643137455F, 0.2823529541F, 0.1003921479F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 2, 4, identity_config(3, 3));
    check(output.size() == expected.size(), "DINOv3 HF resize golden size");
    if (output.size() != expected.size())
        return;
    for (std::size_t i = 0; i < output.size(); ++i)
        check_close(output[i], expected[i], 2e-7F, "DINOv3 HF resize golden value");
}

void test_imagenet_normalization_is_applied_after_resize() {
    const std::vector<float> pixels = {0.485F, 0.456F, 0.406F};
    trtmc::Dinov3PreprocessConfig config;
    config.input_image_h = 1;
    config.input_image_w = 1;
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 1, 1, config);
    check(output.size() == 3, "DINOv3 normalized output size");
    for (float value : output)
        check_close(value, 0.0F, 1e-6F, "DINOv3 ImageNet normalization");
}

void test_invalid_input_is_rejected() {
    bool threw = false;
    try {
        (void)trtmc::preprocess_dinov3_image(nullptr, 0, 0, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DINOv3 empty input rejected");

    threw = false;
    try {
        const std::vector<float> pixels(3, 0.0F);
        auto config = identity_config(1, 1);
        config.image_std[1] = 0.0F;
        (void)trtmc::preprocess_dinov3_image(pixels.data(), 1, 1, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DINOv3 zero std rejected");
}

} // namespace

int main() {
    test_defaults_match_hugging_face_processor();
    test_direct_resize_preserves_rectangular_image_extent();
    test_bilinear_resize_blends_center_pixel();
    test_resize_matches_hugging_face_torchvision_golden();
    test_imagenet_normalization_is_applied_after_resize();
    test_invalid_input_is_rejected();

    if (g_failures != 0) {
        std::cerr << g_failures << " DINOv3 preprocessing test(s) failed\n";
        return 1;
    }
    return 0;
}
