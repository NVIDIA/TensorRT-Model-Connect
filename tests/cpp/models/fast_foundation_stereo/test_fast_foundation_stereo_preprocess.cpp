/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/fast_foundation_stereo/stereo_pipeline.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

void check_close(float actual, float expected, const char* name) {
    if (std::fabs(actual - expected) > 1.0e-6F) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void test_preprocess_matches_rgb_chw_replication_contract() {
    constexpr int32_t height = 700;
    constexpr int32_t width = 700;
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3, 0.0F);
    pixels[0] = 0.25F;
    pixels[1] = 0.5F;
    pixels[2] = 1.0F;
    const auto last = pixels.size() - 3;
    pixels[last] = 0.75F;
    pixels[last + 1] = 0.125F;
    pixels[last + 2] = 0.625F;

    std::vector<float> output;
    trtmc::prepare_fast_foundation_stereo_image(pixels.data(), height, width, output);
    check(output.size() == static_cast<std::size_t>(3) * 704 * 704,
          "stereo preprocess output size");
    if (output.size() != static_cast<std::size_t>(3) * 704 * 704)
        return;

    // Top-left padding replicates source pixel (0,0), then HWC becomes CHW.
    check_close(output[0], 0.25F * 255.0F, "stereo red top-left replicate");
    check_close(output[704 * 704], 0.5F * 255.0F, "stereo green top-left replicate");
    check_close(output[2 * 704 * 704], 255.0F, "stereo blue top-left replicate");

    const auto bottom_right = static_cast<std::size_t>(704) * 704 - 1;
    check_close(output[bottom_right], 0.75F * 255.0F, "stereo red bottom-right replicate");
    check_close(output[704 * 704 + bottom_right], 0.125F * 255.0F,
                "stereo green bottom-right replicate");
}

void test_preprocess_rejects_invalid_input() {
    std::vector<float> output;
    bool null_threw = false;
    try {
        trtmc::prepare_fast_foundation_stereo_image(nullptr, 700, 700, output);
    } catch (const std::invalid_argument&) {
        null_threw = true;
    }
    check(null_threw, "stereo preprocess rejects null image");

    float pixel = 0.0F;
    bool shape_threw = false;
    try {
        trtmc::prepare_fast_foundation_stereo_image(&pixel, 699, 700, output);
    } catch (const std::invalid_argument&) {
        shape_threw = true;
    }
    check(shape_threw, "stereo preprocess rejects wrong shape");
}

} // namespace

int main() {
    test_preprocess_matches_rgb_chw_replication_contract();
    test_preprocess_rejects_invalid_input();
    if (failures == 0)
        std::cout << "All Fast Foundation Stereo preprocess tests passed\n";
    return failures == 0 ? 0 : 1;
}
