/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/owlv2/runtime/image_preprocess_seam.h"

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

constexpr std::int32_t kTarget = 960;

trtmc::Owlv2PreprocessConfig base_config() {
    return {};
}

// 64-pixel checkerboard of 204 and 51, the fixture the expected values below
// were measured on. Both levels are exact in float after the /255 rescale.
std::vector<float> checkerboard(std::int32_t height, std::int32_t width) {
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < width; ++x) {
            const bool on = (((x / 64) + (y / 64)) % 2) == 0;
            const float value = on ? (204.0F / 255.0F) : (51.0F / 255.0F);
            for (std::int32_t c = 0; c < 3; ++c)
                pixels[static_cast<std::size_t>((y * width + x) * 3 + c)] = value;
        }
    }
    return pixels;
}

float at(const std::vector<float>& values, std::int32_t channel, std::int32_t y, std::int32_t x) {
    const auto plane = static_cast<std::size_t>(kTarget) * kTarget;
    return values[static_cast<std::size_t>(channel) * plane +
                  static_cast<std::size_t>(y) * kTarget + x];
}

void test_pad_side_is_the_longer_edge() {
    check(trtmc::compute_owlv2_pad_side(382, 640) == 640, "landscape pads to its width");
    check(trtmc::compute_owlv2_pad_side(640, 382) == 640, "portrait pads to its height");
    check(trtmc::compute_owlv2_pad_side(500, 500) == 500, "a square needs no padding");
}

void test_antialias_sigma_is_zero_when_upsampling() {
    // The reference clamps the sigma at zero and then skips the filter
    // entirely, so an image smaller than the target is never blurred. The test
    // image of this family is 640 wide, which means the family's own E2E could
    // not have caught a wrong answer here.
    check(trtmc::compute_owlv2_antialias_sigma(640, kTarget) == 0.0F, "upsample is not filtered");
    check(trtmc::compute_owlv2_antialias_sigma(960, kTarget) == 0.0F, "no resize is not filtered");
    check_close(trtmc::compute_owlv2_antialias_sigma(1920, kTarget), 0.5F, 1e-6F,
                "a factor of two gives sigma 0.5");
    check_close(trtmc::compute_owlv2_antialias_sigma(2880, kTarget), 1.0F, 1e-6F,
                "a factor of three gives sigma 1.0");
}

void test_downsample_is_anti_aliased() {
    const auto pixels = checkerboard(1920, 1920);
    const auto values = preprocess_owlv2_image(pixels.data(), 1920, 1920, base_config());

    // Measured against the reference processor. The value at a block corner is
    // 0.959654 with the filter and 1.185816 without it, so this one sample
    // separates the two; dropping the filter leaves every other check here
    // passing.
    check_close(at(values, 0, 32, 32), 0.959654F, 2e-3F, "block corner is filtered");
    check_close(at(values, 1, 479, 479), 1.076983F, 2e-3F, "block centre edge is filtered");
    // Deep inside a block the filter changes nothing, which is why a single
    // interior sample cannot stand in for the two above.
    check_close(at(values, 0, 16, 47), -1.047743F, 2e-3F, "block interior is unchanged");
    check_close(at(values, 0, 0, 0), 1.185816F, 2e-3F, "the first pixel is unchanged");
}

void test_padding_is_bottom_right_and_black() {
    const auto pixels = checkerboard(1080, 1920);
    const auto values = preprocess_owlv2_image(pixels.data(), 1080, 1920, base_config());

    // 1080 of 1920 rows carry image, so everything below y = 540 of the
    // resized square is padding, and the padding is zero before normalisation.
    const float black = (0.0F - 0.40821073F) / 0.27577711F;
    check_close(at(values, 2, 959, 959), black, 2e-3F, "the bottom right corner is padding");
    const float black_red = (0.0F - 0.48145466F) / 0.26862954F;
    check_close(at(values, 0, 900, 100), black_red, 2e-3F, "the bottom strip is padding");
    // The top left still carries image, so the padding went to the far side.
    check_close(at(values, 0, 0, 0), 1.185816F, 2e-3F, "the top left is image, not padding");
}

void test_upsample_matches_without_a_filter() {
    const auto pixels = checkerboard(480, 480);
    const auto values = preprocess_owlv2_image(pixels.data(), 480, 480, base_config());
    check_close(at(values, 0, 32, 32), 1.185816F, 2e-3F, "upsampled block is not softened");
    check_close(at(values, 1, 479, 479), 1.309488F, 2e-3F, "upsampled edge is bilinear");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_owlv2_image(nullptr, 16, 16, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    const std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_owlv2_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");
}

} // namespace

int main() {
    test_pad_side_is_the_longer_edge();
    test_antialias_sigma_is_zero_when_upsampling();
    test_downsample_is_anti_aliased();
    test_padding_is_bottom_right_and_black();
    test_upsample_matches_without_a_filter();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "owlv2 image preprocess seam: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
