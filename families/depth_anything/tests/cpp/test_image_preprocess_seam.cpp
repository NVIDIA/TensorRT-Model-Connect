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
    // 8-bit pixel values, as decoded images carry them.
    config.image_mean = {124.0F / 255.0F, 116.0F / 255.0F, 104.0F / 255.0F};
    const auto source = solid_image(4, 4, 124.0F / 255.0F, 116.0F / 255.0F, 104.0F / 255.0F);

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
    const auto source = solid_image(2, 2, 64.0F / 255.0F, 128.0F / 255.0F, 192.0F / 255.0F);

    const auto out = trtmc::preprocess_depth_anything_image(source.data(), 2, 2, config);
    const std::size_t plane = 16U;
    check(std::fabs(out[0] - 64.0F / 255.0F) < 1e-6F, "first plane carries red");
    check(std::fabs(out[plane] - 128.0F / 255.0F) < 1e-6F, "second plane carries green");
    check(std::fabs(out[2U * plane] - 192.0F / 255.0F) < 1e-6F, "third plane carries blue");
}

void test_resize_matches_pillow_bilinear() {
    // Image.fromarray(row).resize(size, Image.BILINEAR) on the 8-bit row 10 200 30 250 90 (one
    // line, gray): to 2x1 gives 103 147, to 3x2 gives 91 117 159 on both lines.
    trtmc::DepthAnythingPreprocessConfig config;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    const float row[] = {10.0F, 200.0F, 30.0F, 250.0F, 90.0F};
    std::vector<float> source;
    for (float value : row)
        source.insert(source.end(), 3U, value / 255.0F);

    config.input_image_h = 1;
    config.input_image_w = 2;
    auto out = trtmc::preprocess_depth_anything_image(source.data(), 1, 5, config);
    check(std::lround(out[0] * 255.0F) == 103 && std::lround(out[1] * 255.0F) == 147,
          "a 5 to 2 downsample matches Pillow's bilinear filter");

    config.input_image_h = 2;
    config.input_image_w = 3;
    out = trtmc::preprocess_depth_anything_image(source.data(), 1, 5, config);
    const long expected[] = {91, 117, 159, 91, 117, 159};
    bool matches = true;
    for (std::size_t index = 0; index < 6U; ++index)
        matches = matches && std::lround(out[index] * 255.0F) == expected[index];
    check(matches, "a 5x1 to 3x2 resample matches Pillow's bilinear filter");
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
    test_resize_matches_pillow_bilinear();
    test_rejects_bad_input();
    if (failures != 0) {
        std::fprintf(stderr, "%d depth_anything preprocess check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("depth_anything preprocess checks passed\n");
    return EXIT_SUCCESS;
}
