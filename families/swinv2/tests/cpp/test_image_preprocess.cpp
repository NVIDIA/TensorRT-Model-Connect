/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/swinv2/runtime/image_preprocess.h"

#include <cmath>
#include <cstddef>
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

trtmc::swinv2::PreprocessConfig identity_config(std::int32_t height, std::int32_t width) {
    trtmc::swinv2::PreprocessConfig config;
    config.input_image_h = height;
    config.input_image_w = width;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    return config;
}

// An RGB source whose bytes follow a fixed formula, as 8-bit values in [0, 1].
std::vector<float> formula_image(std::int32_t height, std::int32_t width) {
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3U);
    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < width; ++x) {
            for (std::int32_t c = 0; c < 3; ++c) {
                const auto byte = (x * 37 + y * 91 + c * 53 + x * y * 13) % 256;
                pixels[(static_cast<std::size_t>(y) * width + x) * 3U + c] =
                    static_cast<float>(byte) / 255.0F;
            }
        }
    }
    return pixels;
}

// Pillow 13.0.0 output of Image.resize((width, height), Resampling.BICUBIC) on
// formula_image(7, 11), interleaved RGB. 4 x 6 shrinks both axes; 9 x 5 grows
// the rows and shrinks the columns, so both filter scales are pinned.
const std::vector<int> kPillow4x6 = {
    65,  120, 162, 135, 147, 149, 128, 155, 166, 150, 136, 118, 104, 93,  105, 82,  124, 186,
    148, 140, 117, 149, 135, 122, 151, 135, 121, 127, 106, 158, 154, 164, 110, 110, 97,  108,
    112, 173, 130, 134, 139, 132, 140, 97,  141, 129, 117, 157, 127, 108, 160, 154, 138, 132,
    103, 161, 150, 111, 122, 187, 173, 122, 138, 127, 115, 135, 152, 140, 135, 133, 176, 91,
};
const std::vector<int> kPillow9x5 = {
    19,  72,  127, 98,  165, 222, 193, 169, 101, 55,  52,  111, 83,  146, 198, 93,  157,
    182, 159, 131, 124, 111, 150, 171, 152, 124, 98,  76,  105, 165, 169, 168, 147, 155,
    133, 112, 116, 123, 153, 171, 168, 113, 115, 82,  121, 171, 121, 83,  125, 151, 149,
    171, 108, 106, 119, 153, 142, 149, 105, 93,  74,  131, 138, 178, 126, 94,  141, 114,
    177, 113, 109, 144, 103, 163, 100, 119, 181, 131, 122, 112, 134, 122, 90,  156, 132,
    95,  145, 149, 119, 158, 132, 191, 123, 150, 112, 166, 119, 148, 127, 149, 108, 172,
    169, 142, 145, 106, 164, 146, 159, 116, 176, 141, 153, 122, 146, 127, 162, 148, 175,
    100, 72,  132, 190, 109, 116, 169, 172, 72,  140, 130, 136, 108, 111, 173, 72,
};

// Pillow 13.0.0 output for formula_image(11, 7) resized to 2 x 6. The height
// loses 9 rows and the width 1 column, so Pillow 13 runs the vertical pass
// first; the horizontal-first order of Pillow 12 differs in 9 of these bytes.
const std::vector<int> kPillow2x6VerticalFirst = {
    101, 146, 128, 128, 148, 145, 136, 138, 137, 154, 115, 147, 140, 146, 138, 140, 104, 149,
    128, 126, 153, 145, 108, 149, 140, 110, 166, 126, 116, 146, 141, 121, 104, 131, 99,  141,
};

void check_against_pillow(std::int32_t source_h, std::int32_t source_w, std::int32_t height,
                          std::int32_t width, const std::vector<int>& golden, const char* name) {
    const auto source = formula_image(source_h, source_w);
    const auto out = trtmc::swinv2::preprocess_image(source.data(), source_h, source_w,
                                                     identity_config(height, width));
    const auto plane = static_cast<std::size_t>(height) * width;
    bool exact = out.size() == 3U * plane && golden.size() == 3U * plane;
    for (std::size_t index = 0; exact && index < plane; ++index) {
        for (std::size_t c = 0; c < 3U; ++c) {
            // Mean 0 and deviation 1 leave the rescaled byte; it must be Pillow's byte.
            const auto byte = static_cast<long>(std::lround(out[c * plane + index] * 255.0F));
            exact = exact && byte == golden[index * 3U + c];
        }
    }
    check(exact, name);
}

void test_bicubic_matches_pillow_byte_for_byte() {
    check_against_pillow(7, 11, 4, 6, kPillow4x6, "a shrink matches Pillow's bicubic bytes");
    check_against_pillow(7, 11, 9, 5, kPillow9x5,
                         "a mixed grow/shrink matches Pillow's bicubic bytes");
    check_against_pillow(11, 7, 2, 6, kPillow2x6VerticalFirst,
                         "a tall shrink runs Pillow 13's vertical pass first");
}

void test_a_wide_image_is_squashed_not_cropped() {
    // The processor resizes straight to the square: no aspect-ratio rule and no
    // crop. The colour boundary of a 64 x 256 half-and-half image must stay in
    // the middle; scaling the short edge and cropping would discard it.
    constexpr std::int32_t kHeight = 64;
    constexpr std::int32_t kWidth = 256;
    std::vector<float> pixels(static_cast<std::size_t>(kHeight) * kWidth * 3U);
    for (std::int32_t y = 0; y < kHeight; ++y) {
        for (std::int32_t x = 0; x < kWidth; ++x) {
            for (std::int32_t c = 0; c < 3; ++c)
                pixels[(static_cast<std::size_t>(y) * kWidth + x) * 3U + c] =
                    x < kWidth / 2 ? 0.0F : 1.0F;
        }
    }
    const auto out =
        trtmc::swinv2::preprocess_image(pixels.data(), kHeight, kWidth, identity_config(32, 32));
    check(out.size() == 3U * 32U * 32U, "the output is planar at the configured size");
    check(out[16U * 32U + 2U] == 0.0F, "the left edge survives the squash");
    check(out[16U * 32U + 29U] == 1.0F, "the right edge survives the squash");
    check(out[16U * 32U + 14U] < 0.5F && out[16U * 32U + 17U] > 0.5F,
          "the boundary stays at the centre");
}

void test_rescale_and_normalisation_follow_the_reference() {
    // The reference rescales a byte in double, rounds to float32, then
    // normalizes per channel in float32.
    trtmc::swinv2::PreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    std::vector<float> pixels(3U * 4U * 4U, 200.0F / 255.0F);
    const auto out = trtmc::swinv2::preprocess_image(pixels.data(), 4, 4, config);
    const auto plane = std::size_t{16};
    for (std::size_t c = 0; c < 3U; ++c) {
        const auto rescaled = static_cast<float>(200.0 * (1.0 / 255.0));
        const float expected = (rescaled - config.image_mean[c]) / config.image_std[c];
        check(out[c * plane] == expected, "each channel uses its own mean and deviation");
    }
}

void test_invalid_input_is_refused() {
    auto config = identity_config(4, 4);
    bool threw = false;
    try {
        trtmc::swinv2::preprocess_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::swinv2::preprocess_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");
}

} // namespace

int main() {
    test_bicubic_matches_pillow_byte_for_byte();
    test_a_wide_image_is_squashed_not_cropped();
    test_rescale_and_normalisation_follow_the_reference();
    test_invalid_input_is_refused();
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "swinv2 image preprocess: all checks passed\n";
    return 0;
}
