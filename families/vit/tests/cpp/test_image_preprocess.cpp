/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vit/runtime/image_preprocess.h"

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

trtmc::VitPreprocessConfig base_config(std::int32_t side) {
    trtmc::VitPreprocessConfig config;
    config.input_image_h = side;
    config.input_image_w = side;
    config.resample = trtmc::VitResample::kBilinear;
    return config;
}

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
    // ViT squashes the whole image onto the square: no aspect-ratio rule and
    // no crop. A 64x256 source is four times wider than tall, and the colour
    // boundary must stay in the middle of the output. Scaling the short edge
    // and cropping - which is what CLIP and DINOv2 do - would throw away
    // three quarters of the width.
    const auto config = base_config(32);
    const auto pixels = half_and_half(64, 256, 0.0F, 1.0F);
    const auto out = trtmc::preprocess_vit_image(pixels.data(), 64, 256, config);
    check(out.size() == 3U * 32U * 32U, "output is CHW at the configured size");

    const auto at = [&](std::int32_t c, std::int32_t y, std::int32_t x) {
        return out[static_cast<std::size_t>(c) * 32U * 32U + static_cast<std::size_t>(y) * 32U + x];
    };
    // mean 0.5, std 0.5: a 0.0 pixel becomes -1 and a 1.0 pixel becomes +1.
    check_close(at(0, 16, 2), -1.0F, 1e-3F, "the left edge survives the squash");
    check_close(at(0, 16, 29), 1.0F, 1e-3F, "the right edge survives the squash");
    check(at(0, 16, 14) < 0.0F && at(0, 16, 17) > 0.0F, "the boundary stays at the centre");
}

void test_bilinear_and_bicubic_are_different_filters() {
    // This is the rule that separates ViT from the other image families here:
    // it resamples BILINEAR where they resample bicubic. The two are close on
    // a flat field and visibly different across an edge, because only the
    // cubic kernel overshoots. Measured on the whole model, swapping them
    // moves the logits by 0.376 and changes the top-5, so the seam must not
    // treat them as interchangeable.
    auto bilinear = base_config(16);
    auto bicubic = base_config(16);
    bicubic.resample = trtmc::VitResample::kBicubic;

    const auto pixels = half_and_half(64, 64, 0.0F, 1.0F);
    const auto a = trtmc::preprocess_vit_image(pixels.data(), 64, 64, bilinear);
    const auto b = trtmc::preprocess_vit_image(pixels.data(), 64, 64, bicubic);
    check(a.size() == b.size(), "both filters produce the same shape");

    float worst = 0.0F;
    for (std::size_t index = 0; index < a.size(); ++index)
        worst = std::max(worst, std::fabs(a[index] - b[index]));
    check(worst > 1e-3F, "bilinear and bicubic are not the same filter");

    // Only the cubic kernel has negative lobes, so only it can overshoot past
    // the input range. The bilinear output is a convex combination and cannot.
    float bilinear_max = -1e9F;
    for (float value : a)
        bilinear_max = std::max(bilinear_max, value);
    check(bilinear_max <= 1.0F + 1e-4F, "bilinear never overshoots its input range");
}

void test_a_flat_image_is_unchanged_by_either_filter() {
    // Both kernels are normalised to sum to one, so a constant field resizes
    // to the same constant. This catches a mis-normalised weight table, which
    // would otherwise show up only as a slight overall brightness shift.
    for (auto resample : {trtmc::VitResample::kBilinear, trtmc::VitResample::kBicubic}) {
        auto config = base_config(8);
        config.resample = resample;
        config.image_mean = {0.0F, 0.0F, 0.0F};
        config.image_std = {1.0F, 1.0F, 1.0F};
        std::vector<float> pixels(static_cast<std::size_t>(40) * 40 * 3U, 0.6F);
        const auto out = trtmc::preprocess_vit_image(pixels.data(), 40, 40, config);
        for (float value : out)
            check_close(value, 0.6F, 3e-3F, "a flat field survives the resize");
    }
}

void test_normalisation_is_per_channel() {
    auto config = base_config(4);
    config.image_mean = {0.0F, 0.5F, 1.0F};
    config.image_std = {1.0F, 0.5F, 2.0F};
    std::vector<float> pixels(3U * 4U * 4U, 1.0F);
    const auto out = trtmc::preprocess_vit_image(pixels.data(), 4, 4, config);
    const auto plane = static_cast<std::size_t>(16);
    check_close(out[0], 1.0F, 1e-5F, "channel 0 uses its own mean and deviation");
    check_close(out[plane], 1.0F, 1e-5F, "channel 1 uses its own mean and deviation");
    check_close(out[2 * plane], 0.0F, 1e-5F, "channel 2 uses its own mean and deviation");
}

void test_invalid_input_is_refused() {
    auto config = base_config(4);
    bool threw = false;
    try {
        trtmc::preprocess_vit_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_vit_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");

    threw = false;
    config = base_config(4);
    config.resample = static_cast<trtmc::VitResample>(0); // NEAREST
    try {
        trtmc::preprocess_vit_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "an unsupported filter is refused rather than approximated");
}

} // namespace

int main() {
    test_a_wide_image_is_squashed_not_cropped();
    test_bilinear_and_bicubic_are_different_filters();
    test_a_flat_image_is_unchanged_by_either_filter();
    test_normalisation_is_per_channel();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "vit image preprocess: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
