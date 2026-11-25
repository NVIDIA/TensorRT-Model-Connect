/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2_with_registers/runtime/image_preprocess.h"

#include <array>
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

void check_close(double actual, double expected, double tolerance, const char* name) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::Dinov2WithRegistersPreprocessConfig base_config() {
    trtmc::Dinov2WithRegistersPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.shortest_edge = 256;
    return config;
}

void test_long_edge_is_floored_not_rounded() {
    // 256 * 640 / 382 is 428.9. The reference truncates, so a 382x640 source
    // resizes to 256x428; rounding to 429 shifts every column of the crop by
    // one and still produces a plausible-looking image.
    const auto config = base_config();
    const auto landscape = trtmc::compute_dinov2_with_registers_resize_shape(382, 640, config);
    check(landscape.height == 256 && landscape.width == 428, "landscape long edge is floored");
    const auto portrait = trtmc::compute_dinov2_with_registers_resize_shape(640, 382, config);
    check(portrait.height == 428 && portrait.width == 256, "portrait long edge is floored");
    const auto square = trtmc::compute_dinov2_with_registers_resize_shape(500, 500, config);
    check(square.height == 256 && square.width == 256, "a square source stays square");
}

void test_a_patch_maps_back_onto_the_source() {
    // This is the rule the Task contract depends on: a patch's footprint is
    // reported in ORIGINAL-image coordinates, so the resize and the centre
    // crop both have to be undone. A 382x640 source resizes to 256x428 and
    // then loses (428-224)/2 = 102 columns from each side, so the crop covers
    // original columns [102/scale, 326/scale] with scale = 428/640.
    const auto config = base_config();
    const auto transform = trtmc::compute_dinov2_with_registers_source_transform(382, 640, config);
    check_close(transform.scale_x, 428.0 / 640.0, 1e-12, "x scale is the resize ratio");
    check_close(transform.scale_y, 256.0 / 382.0, 1e-12, "y scale is the resize ratio");
    check_close(transform.offset_x, -102.0, 1e-12, "x offset is the floored crop");
    check_close(transform.offset_y, -16.0, 1e-12, "y offset is the floored crop");

    // Patch column 0 starts at processed x=0, which is original x=102/scale.
    const double left = (0.0 - transform.offset_x) / transform.scale_x;
    check_close(left, 102.0 * 640.0 / 428.0, 1e-9, "the first patch starts inside the source");
    // The last patch of a 16-wide grid ends at processed x=224.
    const double right = (224.0 - transform.offset_x) / transform.scale_x;
    check_close(right, 326.0 * 640.0 / 428.0, 1e-9, "the last patch ends inside the source");
    check(left > 0.0 && right < 640.0, "the crop lies strictly inside a wide source");

    // A square source loses nothing on either axis, so the crop spans it all.
    const auto square = trtmc::compute_dinov2_with_registers_source_transform(500, 500, config);
    const double square_left = (0.0 - square.offset_x) / square.scale_x;
    const double square_right = (224.0 - square.offset_x) / square.scale_x;
    check_close(square_left, 16.0 * 500.0 / 256.0, 1e-9, "a square source crops symmetrically");
    check_close(square_right, 240.0 * 500.0 / 256.0, 1e-9, "a square source crops symmetrically");
}

void test_normalisation_is_per_channel() {
    auto config = base_config();
    config.input_image_h = 4;
    config.input_image_w = 4;
    config.shortest_edge = 4;
    config.image_mean = {0.0F, 0.5F, 1.0F};
    config.image_std = {1.0F, 0.5F, 2.0F};
    std::vector<float> pixels(3U * 4U * 4U, 1.0F);
    const auto out = trtmc::preprocess_dinov2_with_registers_image(pixels.data(), 4, 4, config);
    const auto plane = static_cast<std::size_t>(16);
    check_close(out[0], 1.0, 1e-5, "channel 0 uses its own mean and deviation");
    check_close(out[plane], 1.0, 1e-5, "channel 1 uses its own mean and deviation");
    check_close(out[2 * plane], 0.0, 1e-5, "channel 2 uses its own mean and deviation");
}

void test_nonconstant_bicubic_matches_pillow() {
    // Independent RGB fixtures from Pillow Image.resize(BICUBIC), then a
    // floored center crop. Cover enlargement, antialiased reduction, and
    // odd crop offsets without using this implementation as the oracle.
    auto run = [](int height, int width, int shortest,
                  const std::array<unsigned char, 48>& expected) {
        auto config = base_config();
        config.input_image_h = config.input_image_w = 4;
        config.shortest_edge = shortest;
        config.image_mean = {0, 0, 0};
        config.image_std = {1, 1, 1};
        std::vector<float> input(static_cast<std::size_t>(height) * width * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    input[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                        static_cast<float>((y * 71 + x * 29 + c * 97) % 256) / 255.0F;
        const auto actual =
            trtmc::preprocess_dinov2_with_registers_image(input.data(), height, width, config);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                for (int c = 0; c < 3; ++c)
                    check_close(actual[c * 16 + y * 4 + x], expected[(y * 4 + x) * 3 + c] / 255.0,
                                1e-7, "bicubic crop matches independent Pillow fixture");
    };
    run(3, 5, 7,
        {
            38,  137, 209, 51,  150, 235, 64,  163, 234, 77,  178, 145, 67,  182, 118, 80,
            195, 138, 93,  207, 144, 105, 229, 107, 103, 200, 41,  116, 213, 54,  129, 226,
            67,  142, 254, 80,  139, 119, 60,  152, 130, 72,  165, 149, 86,  185, 172, 106,
        });
    run(9, 13, 5,
        {
            105, 192, 119, 155, 152, 91,  179, 77,  154, 112, 107, 178, 148, 95,  158, 77,
            123, 148, 100, 151, 150, 125, 143, 93,  98,  176, 108, 154, 143, 88,  163, 82,
            141, 99,  91,  160, 134, 114, 147, 102, 117, 181, 92,  145, 133, 133, 168, 84,
        });
    run(5, 7, 5,
        {
            29,  126, 223, 58,  155, 252, 87,  184, 25,  116, 213, 54,  100, 197, 38,  129,
            226, 67,  158, 255, 96,  187, 28,  125, 171, 12,  109, 200, 41,  138, 229, 70,
            167, 2,   99,  196, 242, 83,  180, 15,  112, 209, 44,  141, 238, 73,  170, 11,
        });
}

void test_panorama_only_retains_the_center_crop() {
    auto config = base_config();
    config.image_mean = {0, 0, 0};
    config.image_std = {1, 1, 1};
    std::vector<float> input(10000U * 3U, 0.5F);
    const auto actual =
        trtmc::preprocess_dinov2_with_registers_image(input.data(), 1, 10000, config);
    check(actual.size() == 224U * 224U * 3U, "panorama returns only the requested crop");
    for (float value : actual)
        check_close(value, 128.0 / 255.0, 1e-7,
                    "panorama keeps the full-image sampling coordinates");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_dinov2_with_registers_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.shortest_edge = 100; // smaller than the 224 crop
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_dinov2_with_registers_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a resize smaller than the crop is refused");

    threw = false;
    config = base_config();
    config.image_std = {1.0F, 0.0F, 1.0F};
    try {
        trtmc::preprocess_dinov2_with_registers_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");

    threw = false;
    config = base_config();
    config.input_image_w = 192; // not square
    try {
        trtmc::compute_dinov2_with_registers_resize_shape(4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a non-square crop is refused");
}

} // namespace

int main() {
    test_long_edge_is_floored_not_rounded();
    test_a_patch_maps_back_onto_the_source();
    test_normalisation_is_per_channel();
    test_nonconstant_bicubic_matches_pillow();
    test_panorama_only_retains_the_center_crop();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "dinov2_with_registers image preprocess: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
