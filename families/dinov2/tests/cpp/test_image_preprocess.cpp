/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/image_preprocess.h"

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
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::Dinov2PreprocessConfig base_config() {
    trtmc::Dinov2PreprocessConfig config;
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
    const auto landscape = trtmc::compute_dinov2_resize_shape(382, 640, config);
    check(landscape.height == 256 && landscape.width == 428, "landscape long edge is floored");
    const auto portrait = trtmc::compute_dinov2_resize_shape(640, 382, config);
    check(portrait.height == 428 && portrait.width == 256, "portrait long edge is floored");
    const auto square = trtmc::compute_dinov2_resize_shape(500, 500, config);
    check(square.height == 256 && square.width == 256, "a square source stays square");
}

void test_a_patch_maps_back_onto_the_source() {
    // This is the rule the Task contract depends on: a patch's footprint is
    // reported in ORIGINAL-image coordinates, so the resize and the centre
    // crop both have to be undone. A 382x640 source resizes to 256x428 and
    // then loses (428-224)/2 = 102 columns from each side, so the crop covers
    // original columns [102/scale, 326/scale] with scale = 428/640.
    const auto config = base_config();
    const auto transform = trtmc::compute_dinov2_source_transform(382, 640, config);
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
    const auto square = trtmc::compute_dinov2_source_transform(500, 500, config);
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
    const auto out = trtmc::preprocess_dinov2_image(pixels.data(), 4, 4, config);
    const auto plane = static_cast<std::size_t>(16);
    check_close(out[0], 1.0, 1e-5, "channel 0 uses its own mean and deviation");
    check_close(out[plane], 1.0, 1e-5, "channel 1 uses its own mean and deviation");
    check_close(out[2 * plane], 0.0, 1e-5, "channel 2 uses its own mean and deviation");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_dinov2_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.shortest_edge = 100; // smaller than the 224 crop
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_dinov2_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a resize smaller than the crop is refused");

    threw = false;
    config = base_config();
    config.image_std = {1.0F, 0.0F, 1.0F};
    try {
        trtmc::preprocess_dinov2_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");

    threw = false;
    config = base_config();
    config.input_image_w = 192; // not square
    try {
        trtmc::compute_dinov2_resize_shape(4, 4, config);
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
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "dinov2 image preprocess: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
