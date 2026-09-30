/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything/runtime/image_preprocess_seam.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<float> solid_image(int32_t height, int32_t width, float r, float g, float b) {
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3U);
    for (std::size_t i = 0; i < pixels.size(); i += 3U) {
        pixels[i] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
    return pixels;
}

void test_output_is_chw_and_normalized() {
    trtmc::DepthAnythingPreprocessConfig config;
    config.input_image_h = 8;
    config.input_image_w = 16;
    const auto source = solid_image(4, 4, 0.485F, 0.456F, 0.406F);

    const auto out = trtmc::preprocess_depth_anything_image(source.data(), 4, 4, config);
    check(out.size() == 3U * 8U * 16U, "output covers three planes of the engine input size");

    // A solid image at exactly the mean normalizes to zero everywhere.
    for (float value : out)
        check(std::fabs(value) < 1e-5F, "mean-valued pixels normalize to zero");
}

void test_channels_are_separated_into_planes() {
    trtmc::DepthAnythingPreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    const auto source = solid_image(2, 2, 0.25F, 0.5F, 0.75F);

    const auto out = trtmc::preprocess_depth_anything_image(source.data(), 2, 2, config);
    const std::size_t plane = 16U;
    check(std::fabs(out[0] - 0.25F) < 1e-4F, "first plane carries red");
    check(std::fabs(out[plane] - 0.5F) < 1e-4F, "second plane carries green");
    check(std::fabs(out[2U * plane] - 0.75F) < 1e-4F, "third plane carries blue");
}

void test_rejects_bad_input() {
    trtmc::DepthAnythingPreprocessConfig config;
    const auto source = solid_image(2, 2, 0.1F, 0.1F, 0.1F);
    bool threw = false;
    try {
        trtmc::preprocess_depth_anything_image(nullptr, 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is rejected");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    try {
        trtmc::preprocess_depth_anything_image(source.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is rejected");
}

} // namespace

int main() {
    test_output_is_chw_and_normalized();
    test_channels_are_separated_into_planes();
    test_rejects_bad_input();
    if (failures != 0) {
        std::fprintf(stderr, "%d depth_anything preprocess check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("depth_anything preprocess checks passed\n");
    return EXIT_SUCCESS;
}
