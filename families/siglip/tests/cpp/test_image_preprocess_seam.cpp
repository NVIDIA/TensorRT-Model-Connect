/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip/runtime/image_preprocess_seam.h"

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

trtmc::SiglipPreprocessConfig base_config(std::int32_t side) {
    trtmc::SiglipPreprocessConfig config;
    config.input_image_h = side;
    config.input_image_w = side;
    return config;
}

// A source whose left half is one colour and right half another. After a
// squash both halves are still half the output, whatever the aspect ratio
// was; after an aspect-preserving resize and a centre crop they would not be.
std::vector<float> half_and_half(std::int32_t height, std::int32_t width, float left, float right) {
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3U);
    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < width; ++x) {
            const float value = x < width / 2 ? left : right;
            for (std::int32_t c = 0; c < 3; ++c)
                pixels[(static_cast<std::size_t>(y) * width + x) * 3U + c] = value;
        }
    }
    return pixels;
}

void test_a_wide_image_is_squashed_not_cropped() {
    // This is the rule that separates SigLIP from CLIP. A 64x256 source is
    // four times wider than tall. SigLIP squashes the whole thing onto the
    // square, so the colour boundary stays in the middle of the output. CLIP
    // would scale the short edge and crop the sides, which would throw away
    // three quarters of the width and move nothing in the output at all.
    const auto config = base_config(32);
    const auto pixels = half_and_half(64, 256, 0.0F, 1.0F);
    const auto out = trtmc::preprocess_siglip_image(pixels.data(), 64, 256, config);
    check(out.size() == 3U * 32U * 32U, "output is CHW at the configured size");

    const auto at = [&](std::int32_t c, std::int32_t y, std::int32_t x) {
        return out[static_cast<std::size_t>(c) * 32U * 32U + static_cast<std::size_t>(y) * 32U + x];
    };
    // mean 0.5, std 0.5: a 0.0 pixel becomes -1 and a 1.0 pixel becomes +1.
    check_close(at(0, 16, 2), -1.0F, 1e-3F, "the left edge survives the squash");
    check_close(at(0, 16, 29), 1.0F, 1e-3F, "the right edge survives the squash");
    // The boundary is at the centre column, which it would not be if the
    // sides had been cropped away.
    check(at(0, 16, 14) < 0.0F && at(0, 16, 17) > 0.0F, "the boundary stays at the centre");
}

void test_a_tall_image_is_squashed_too() {
    const auto config = base_config(16);
    std::vector<float> pixels(static_cast<std::size_t>(256) * 64 * 3U);
    for (std::int32_t y = 0; y < 256; ++y) {
        for (std::int32_t x = 0; x < 64; ++x) {
            const float value = y < 128 ? 0.0F : 1.0F;
            for (std::int32_t c = 0; c < 3; ++c)
                pixels[(static_cast<std::size_t>(y) * 64 + x) * 3U + c] = value;
        }
    }
    const auto out = trtmc::preprocess_siglip_image(pixels.data(), 256, 64, config);
    const auto at = [&](std::int32_t y) { return out[static_cast<std::size_t>(y) * 16U + 8U]; };
    check_close(at(1), -1.0F, 1e-3F, "the top survives the squash");
    check_close(at(14), 1.0F, 1e-3F, "the bottom survives the squash");
}

void test_normalisation_is_per_channel() {
    auto config = base_config(4);
    config.image_mean = {0.0F, 0.5F, 1.0F};
    config.image_std = {1.0F, 0.5F, 2.0F};
    std::vector<float> pixels(3U * 4U * 4U, 1.0F);
    const auto out = trtmc::preprocess_siglip_image(pixels.data(), 4, 4, config);
    const auto plane = static_cast<std::size_t>(16);
    check_close(out[0], 1.0F, 1e-5F, "channel 0 uses its own mean and deviation");
    check_close(out[plane], 1.0F, 1e-5F, "channel 1 uses its own mean and deviation");
    check_close(out[2 * plane], 0.0F, 1e-5F, "channel 2 uses its own mean and deviation");
}

void test_invalid_input_is_refused() {
    auto config = base_config(4);
    bool threw = false;
    try {
        trtmc::preprocess_siglip_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_siglip_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");

    threw = false;
    config.image_std = {1.0F, 1.0F, 1.0F};
    config.image_mean = {0.0F, 0.0F};
    try {
        trtmc::preprocess_siglip_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a two-channel mean is refused");
}

} // namespace

int main() {
    test_a_wide_image_is_squashed_not_cropped();
    test_a_tall_image_is_squashed_too();
    test_normalisation_is_per_channel();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "siglip image preprocess seam: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
