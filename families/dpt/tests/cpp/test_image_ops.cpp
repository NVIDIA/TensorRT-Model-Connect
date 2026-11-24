/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dpt/runtime/image_ops.h"
#include "families/dpt/tests/cpp/image_ops_goldens.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// The recurrence of generate_image_ops_goldens.py.
std::vector<std::uint8_t> lcg_image(int height, int width, std::uint32_t seed) {
    std::vector<std::uint8_t> values(static_cast<std::size_t>(height) * width * 3U);
    std::uint32_t state = seed;
    for (auto& value : values) {
        state = (state * 1103515245U + 12345U) & 0x7FFFFFFFU;
        value = static_cast<std::uint8_t>((state >> 16) & 0xFFU);
    }
    return values;
}

// What the CLI hands a family: 8-bit samples divided by 255.
std::vector<float> to_unit(const std::vector<std::uint8_t>& bytes) {
    std::vector<float> pixels(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index)
        pixels[index] = static_cast<float>(bytes[index]) / 255.0F;
    return pixels;
}

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_bilinear_resize_matches_pillow() {
    for (const auto& golden : dpt_goldens::kResizeGoldens) {
        const auto source = lcg_image(golden.height, golden.width, golden.seed);
        const auto resized = trtmc::dpt::resize_bilinear_rgb8(
            source.data(), golden.height, golden.width, golden.out_height, golden.out_width);
        check(resized == golden.expected, "8-bit bilinear resize is byte-identical to Pillow");
    }
}

void test_rgb8_recovers_every_level_and_clamps() {
    std::vector<float> levels(256U * 3U);
    for (std::size_t level = 0; level < 256U; ++level) {
        for (std::size_t channel = 0; channel < 3U; ++channel)
            levels[level * 3U + channel] = static_cast<float>(level) / 255.0F;
    }
    const auto rgb = trtmc::dpt::to_rgb8(levels.data(), 16, 16);
    bool exact = true;
    for (std::size_t index = 0; index < rgb.size(); ++index)
        exact = exact && rgb[index] == index / 3U;
    check(exact, "every 8-bit level survives the round trip through value / 255");

    const float outside[3] = {-0.25F, 1.5F, 0.5F};
    const auto clamped = trtmc::dpt::to_rgb8(outside, 1, 1);
    check(clamped[0] == 0 && clamped[1] == 255 && clamped[2] == 128,
          "values outside [0, 1] clamp to the 8-bit range");

    const float invalid[3] = {0.5F, std::numeric_limits<float>::quiet_NaN(), 0.5F};
    check(throws_invalid_argument([&] { trtmc::dpt::to_rgb8(invalid, 1, 1); }),
          "a non-finite pixel is rejected");
}

void test_preprocessing_matches_the_image_processor() {
    const auto& size = dpt_goldens::kProcessorSize;
    const auto source = to_unit(lcg_image(size[0], size[1], dpt_goldens::kProcessorSeed));
    trtmc::dpt::PreprocessConfig config;
    config.input_image_h = size[2];
    config.input_image_w = size[3];
    config.rescale_factor = 0.00392156862745098;
    const auto pixel_values = trtmc::dpt::preprocess_image(source.data(), size[0], size[1], config);
    check(pixel_values.size() == dpt_goldens::kProcessorGolden.size() &&
              std::memcmp(pixel_values.data(), dpt_goldens::kProcessorGolden.data(),
                          pixel_values.size() * sizeof(float)) == 0,
          "resize, rescale and normalize match DPTImageProcessor bit for bit");
}

void test_bicubic_resize_matches_torch() {
    const auto& size = dpt_goldens::kBicubicSourceSize;
    float range = 0.0F;
    for (const float value : dpt_goldens::kBicubicSource)
        range = std::fmax(range, std::fabs(value));
    // torch's CPU kernel fuses some multiply-adds, as its compiler chooses; this
    // one evaluates the same formula unfused. Sixteen taps whose weights sum to
    // at most 1.6 in magnitude stay within 32 float32 epsilons of the range.
    const float tolerance = 32.0F * std::numeric_limits<float>::epsilon() * range;
    for (const auto& golden : dpt_goldens::kBicubicGoldens) {
        const auto resized = trtmc::dpt::resize_bicubic(dpt_goldens::kBicubicSource.data(), size[0],
                                                        size[1], golden.height, golden.width);
        bool close = resized.size() == golden.expected.size();
        for (std::size_t index = 0; close && index < resized.size(); ++index)
            close = std::fabs(resized[index] - golden.expected[index]) <= tolerance;
        check(close, "bicubic resize matches torch interpolate(align_corners=False)");
    }
    const auto same = trtmc::dpt::resize_bicubic(dpt_goldens::kBicubicSource.data(), size[0],
                                                 size[1], size[0], size[1]);
    check(same == dpt_goldens::kBicubicSource, "a same-size bicubic resize is the identity");
}

void test_rejects_bad_input() {
    const std::uint8_t byte = 0;
    const float value = 0.0F;
    trtmc::dpt::PreprocessConfig config;
    check(throws_invalid_argument([&] { trtmc::dpt::to_rgb8(nullptr, 1, 1); }),
          "a null image is rejected");
    check(throws_invalid_argument([&] { trtmc::dpt::to_rgb8(&value, 0, 1); }),
          "an empty image is rejected");
    check(throws_invalid_argument([&] { trtmc::dpt::resize_bilinear_rgb8(&byte, 1, 1, 0, 1); }),
          "an empty resize target is rejected");
    check(throws_invalid_argument([&] { trtmc::dpt::resize_bicubic(nullptr, 1, 1, 1, 1); }),
          "a null depth map is rejected");
    config.image_std = {0.5F, 0.0F, 0.5F};
    const float pixel[3] = {0.5F, 0.5F, 0.5F};
    check(throws_invalid_argument([&] { trtmc::dpt::preprocess_image(pixel, 1, 1, config); }),
          "a zero standard deviation is rejected");
    // Finite and non-zero, but 0.5 / 1e-40 overflows float32.
    config.image_std = {0.5F, 1e-40F, 0.5F};
    check(throws_invalid_argument([&] { trtmc::dpt::preprocess_image(pixel, 1, 1, config); }),
          "a standard deviation that normalizes a sample to infinity is rejected");
    config.image_std = {0.5F, 0.5F, 0.5F};
    config.rescale_factor = 1e308;
    check(throws_invalid_argument([&] { trtmc::dpt::preprocess_image(pixel, 1, 1, config); }),
          "a rescale factor that takes 255 past float32 is rejected");
}

} // namespace

int main() {
    test_bilinear_resize_matches_pillow();
    test_rgb8_recovers_every_level_and_clamps();
    test_preprocessing_matches_the_image_processor();
    test_bicubic_resize_matches_torch();
    test_rejects_bad_input();
    if (failures != 0) {
        std::fprintf(stderr, "%d dpt image check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("dpt image checks passed\n");
    return EXIT_SUCCESS;
}
