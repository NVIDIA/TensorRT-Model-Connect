/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_vit/runtime/image_preprocess_seam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The reference below must compute its resize weights as the runtime file does: without
// fused multiply-adds. That file includes stb_image_resize2.h, which turns contraction
// off for the code after it; this is the same guard.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#else
#pragma STDC FP_CONTRACT OFF
#endif

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void test_bilinear_resize() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = 3;
    config.input_image_w = 3;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto values = trtmc::preprocess_timm_vit_image(pixels.data(), 2, 2, config);
    check(values.size() == 27, "preprocess size");
    if (values.size() == 27)
        check_close(values[4], 0.5F, 1e-6F, "bilinear center blend");
}

void test_normalization() {
    const std::vector<float> pixels(12U, 0.75F);
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.crop_pct = 1.0F;
    config.interpolation = "bilinear";
    config.image_mean = {0.25F, 0.5F, 0.75F};
    config.image_std = {0.5F, 0.25F, 0.125F};

    const auto values = trtmc::preprocess_timm_vit_image(pixels.data(), 2, 2, config);
    check_close(values[0], 1.0F, 1e-6F, "red normalization");
    check_close(values[4], 1.0F, 1e-6F, "green normalization");
    check_close(values[8], 0.0F, 1e-6F, "blue normalization");
}

void test_torchvision_short_edge_geometry() {
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.crop_pct = 0.9F;

    const auto landscape = trtmc::compute_timm_vit_resize_shape(320, 426, config);
    check(landscape.height == 248, "landscape short edge");
    check(landscape.width == 330, "landscape aspect ratio");
    const auto portrait = trtmc::compute_timm_vit_resize_shape(426, 320, config);
    check(portrait.height == 330, "portrait aspect ratio");
    check(portrait.width == 248, "portrait short edge");
}

// The previous whole-image, per-element implementation of the bicubic path, kept as the
// oracle for the cropped, fused passes: Pillow's 8-bit bicubic resize of the whole image,
// then torchvision's center crop and the normalization. The outputs must agree bit for bit.
constexpr std::int32_t kReferencePrecisionBits = 22;

struct ReferenceSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double reference_cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<ReferenceSpan> reference_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<ReferenceSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                reference_cubic((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        for (double weight : floating) {
            const double scaled =
                weight / total * static_cast<double>(std::int64_t{1} << kReferencePrecisionBits);
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

std::uint8_t reference_tap_sum(const std::uint8_t* source, std::int32_t stride,
                               const ReferenceSpan& span) {
    std::int64_t sum = std::int64_t{1} << (kReferencePrecisionBits - 1);
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        sum += static_cast<std::int64_t>(
                   source[(span.first + static_cast<std::int32_t>(index)) * stride]) *
               span.weights[index];
    }
    if (sum <= 0)
        return 0;
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kReferencePrecisionBits, 255));
}

std::int32_t reference_crop_offset(std::int32_t resized, std::int32_t target) {
    const std::int32_t difference = resized - target;
    const std::int32_t half = difference / 2;
    return (difference % 2 != 0 && half % 2 != 0) ? half + 1 : half;
}

std::vector<float> reference_bicubic_preprocess(const std::vector<float>& source,
                                                std::int32_t input_h, std::int32_t input_w,
                                                const trtmc::TimmVitPreprocessConfig& config) {
    const auto shape = trtmc::compute_timm_vit_resize_shape(input_h, input_w, config);
    const std::int32_t resized_h = shape.height;
    const std::int32_t resized_w = shape.width;

    std::vector<std::uint8_t> input(source.size());
    std::transform(source.begin(), source.end(), input.begin(), [](float value) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    });

    const auto horizontal_plan = reference_plan(input_w, resized_w);
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(input_h) * resized_w * 3U);
    for (std::int32_t y = 0; y < input_h; ++y)
        for (std::int32_t x = 0; x < resized_w; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
                horizontal[(static_cast<std::size_t>(y) * resized_w + x) * 3U + c] =
                    reference_tap_sum(input.data() + static_cast<std::size_t>(y) * input_w * 3U + c,
                                      3, horizontal_plan[static_cast<std::size_t>(x)]);

    const auto vertical_plan = reference_plan(input_h, resized_h);
    std::vector<float> resized(static_cast<std::size_t>(resized_h) * resized_w * 3U);
    for (std::int32_t y = 0; y < resized_h; ++y)
        for (std::int32_t x = 0; x < resized_w; ++x)
            for (std::int32_t c = 0; c < 3; ++c)
                resized[(static_cast<std::size_t>(y) * resized_w + x) * 3U + c] =
                    static_cast<float>(reference_tap_sum(
                        horizontal.data() + static_cast<std::size_t>(x) * 3U + c, resized_w * 3,
                        vertical_plan[static_cast<std::size_t>(y)])) /
                    255.0F;

    const std::int32_t crop_y = reference_crop_offset(resized_h, config.input_image_h);
    const std::int32_t crop_x = reference_crop_offset(resized_w, config.input_image_w);
    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::int32_t y = 0; y < config.input_image_h; ++y)
        for (std::int32_t x = 0; x < config.input_image_w; ++x)
            for (std::size_t c = 0; c < 3; ++c)
                pixel_values[c * plane + static_cast<std::size_t>(y) * config.input_image_w + x] =
                    (resized[(static_cast<std::size_t>(crop_y + y) * resized_w + crop_x + x) * 3U +
                             c] -
                     config.image_mean[c]) /
                    config.image_std[c];
    return pixel_values;
}

// A fixed pseudo-random image: mostly k/255 values, as a decoded 8-bit image holds,
// mixed with arbitrary values, values out of [0, 1] and values next to a rounding half.
std::vector<float> make_image(std::int32_t height, std::int32_t width, std::uint32_t seed) {
    std::uint32_t state = seed * 2654435761U + 12345U;
    const auto next = [&state]() {
        state = state * 1664525U + 1013904223U;
        return state >> 8;
    };
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3U);
    for (float& pixel : pixels) {
        const std::uint32_t kind = next() % 16U;
        const auto level = static_cast<float>(next() % 256U);
        if (kind < 10U) {
            pixel = level / 255.0F;
        } else if (kind < 13U) {
            pixel = static_cast<float>(next() % 100000U) / 100000.0F;
        } else if (kind == 13U) {
            pixel = static_cast<float>(next() % 4000U) / 1000.0F - 1.5F;
        } else {
            const float half = (level + 0.5F) / 255.0F;
            pixel = kind == 14U ? std::nextafter(half, 0.0F) : std::nextafter(half, 2.0F);
        }
    }
    return pixels;
}

bool same_bits(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

void check_bicubic_matches_reference(std::int32_t input_h, std::int32_t input_w,
                                     std::int32_t output_h, std::int32_t output_w, float crop_pct,
                                     std::uint32_t seed) {
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = output_h;
    config.input_image_w = output_w;
    config.crop_pct = crop_pct;
    config.interpolation = "bicubic";
    config.image_mean = {0.485F, 0.456F, 0.406F};
    config.image_std = {0.229F, 0.224F, 0.225F};
    const auto pixels = make_image(input_h, input_w, seed);
    const auto expected = reference_bicubic_preprocess(pixels, input_h, input_w, config);
    const auto actual = trtmc::preprocess_timm_vit_image(pixels.data(), input_h, input_w, config);
    if (!same_bits(actual, expected)) {
        std::cerr << "FAIL: bicubic preprocess differs from the whole-image reference for "
                  << input_h << "x" << input_w << " -> " << output_h << "x" << output_w
                  << " crop_pct=" << crop_pct << '\n';
        ++failures;
    }
}

void test_bicubic_matches_whole_image_reference() {
    // The benchmark request: a 640x382 image, downscaled and cropped from 415x248.
    check_bicubic_matches_reference(382, 640, 224, 224, 0.9F, 1U);
    // Portrait, upscale, a square source, no crop margin, and an identity resize.
    check_bicubic_matches_reference(640, 382, 224, 224, 0.9F, 2U);
    check_bicubic_matches_reference(60, 80, 224, 224, 0.9F, 3U);
    check_bicubic_matches_reference(300, 300, 224, 224, 0.875F, 4U);
    check_bicubic_matches_reference(500, 333, 224, 224, 1.0F, 5U);
    check_bicubic_matches_reference(224, 224, 224, 224, 1.0F, 6U);
    // Odd crop margins on both axes, the smallest sources and a strong downscale.
    check_bicubic_matches_reference(37, 53, 16, 16, 0.9F, 7U);
    check_bicubic_matches_reference(1, 1, 8, 8, 0.9F, 8U);
    check_bicubic_matches_reference(2, 3, 5, 5, 0.7F, 9U);
    check_bicubic_matches_reference(1, 400, 4, 4, 1.0F, 10U);
    check_bicubic_matches_reference(1200, 1600, 224, 224, 0.9F, 11U);
    // A rectangular model input takes the other resize-shape branch.
    check_bicubic_matches_reference(382, 640, 192, 256, 0.9F, 12U);
    check_bicubic_matches_reference(640, 382, 256, 192, 0.95F, 13U);
    // A seeded sweep of small geometries.
    std::uint32_t state = 20261116U;
    const auto next = [&state](std::uint32_t modulus) {
        state = state * 1664525U + 1013904223U;
        return static_cast<std::int32_t>((state >> 8) % modulus);
    };
    for (std::uint32_t round = 0; round < 60U; ++round) {
        const std::int32_t input_h = 1 + next(90U);
        const std::int32_t input_w = 1 + next(90U);
        const std::int32_t output_h = 1 + next(40U);
        const std::int32_t output_w = round % 3U == 0U ? 1 + next(40U) : output_h;
        const float crop_pct = 0.5F + static_cast<float>(next(51U)) / 100.0F;
        check_bicubic_matches_reference(input_h, input_w, output_h, output_w, crop_pct,
                                        100U + round);
    }
}

void test_bicubic_special_values() {
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = 6;
    config.input_image_w = 6;
    config.crop_pct = 0.9F;
    config.interpolation = "bicubic";
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> pixels = make_image(9, 11, 77U);
    pixels[0] = infinity;
    pixels[4] = -infinity;
    pixels[8] = -0.0F;
    pixels[12] = std::numeric_limits<float>::denorm_min();
    pixels[16] = std::numeric_limits<float>::max();
    pixels[20] = std::nextafter(0.5F / 255.0F, 0.0F);
    pixels[24] = 0.5F / 255.0F;
    const auto expected = reference_bicubic_preprocess(pixels, 9, 11, config);
    const auto actual = trtmc::preprocess_timm_vit_image(pixels.data(), 9, 11, config);
    check(same_bits(actual, expected), "bicubic special values match the reference");

    // A NaN sample reads as 0: the image with the NaN equals the image with a zero there.
    std::vector<float> with_nan = pixels;
    std::vector<float> with_zero = pixels;
    for (std::size_t index = 30; index < with_nan.size(); index += 41) {
        with_nan[index] = std::numeric_limits<float>::quiet_NaN();
        with_zero[index] = 0.0F;
    }
    const auto from_nan = trtmc::preprocess_timm_vit_image(with_nan.data(), 9, 11, config);
    const auto from_zero = trtmc::preprocess_timm_vit_image(with_zero.data(), 9, 11, config);
    check(same_bits(from_nan, from_zero), "a NaN sample reads as zero");
    check(!same_bits(from_zero, actual), "the NaN samples feed the output");
}

// The resize weights are computed without fused multiply-adds, in the runtime file and in
// the reference above. For 3315 source rows resized to 281, a fused multiply-add rounds one
// weight of output row 55 to -16765 where the separate operations give -16766. That row
// reads 47 source rows from row 631 on; these levels put its sum next to a byte boundary,
// so the byte is 114 with the right weight and 115 with the fused one.
void test_bicubic_weights_use_no_fused_multiply_add() {
    constexpr std::int32_t kHeight = 3315;
    constexpr std::int32_t kWidth = 100;
    constexpr std::int32_t kFirstRow = 631;
    constexpr std::int32_t kOutputRow = 55;
    constexpr std::uint8_t kRowLevels[] = {
        94,  60,  243, 112, 85,  248, 85, 105, 152, 32,  228, 217, 93,  88,  108, 3,
        24,  133, 8,   107, 123, 202, 96, 214, 124, 139, 135, 71,  78,  108, 209, 2,
        101, 82,  210, 139, 117, 96,  95, 166, 115, 250, 139, 88,  146, 196, 170,
    };
    std::vector<float> pixels(static_cast<std::size_t>(kHeight) * kWidth * 3U, 128.0F / 255.0F);
    std::int32_t row = kFirstRow;
    for (const std::uint8_t level : kRowLevels) {
        float* begin = pixels.data() + static_cast<std::size_t>(row) * kWidth * 3U;
        std::fill(begin, begin + kWidth * 3, static_cast<float>(level) / 255.0F);
        ++row;
    }
    trtmc::TimmVitPreprocessConfig config;
    config.input_image_h = 281;
    config.input_image_w = 8;
    config.crop_pct = 1.0F;
    config.interpolation = "bicubic";
    config.image_mean = {0.5F, 0.5F, 0.5F};
    config.image_std = {0.5F, 0.5F, 0.5F};
    const auto expected = reference_bicubic_preprocess(pixels, kHeight, kWidth, config);
    const auto actual = trtmc::preprocess_timm_vit_image(pixels.data(), kHeight, kWidth, config);
    check(same_bits(actual, expected), "bicubic weights match the reference for 3315 -> 281 rows");
    const auto index = static_cast<std::size_t>(kOutputRow) * config.input_image_w;
    check(actual.size() > index && actual[index] == (114.0F / 255.0F - 0.5F) / 0.5F,
          "bicubic weights use no fused multiply-add");
}

void test_invalid_interpolation() {
    bool threw = false;
    try {
        const std::vector<float> pixels(12U, 0.0F);
        trtmc::TimmVitPreprocessConfig config;
        config.input_image_h = 2;
        config.input_image_w = 2;
        config.crop_pct = 1.0F;
        config.interpolation = "nearest";
        (void)trtmc::preprocess_timm_vit_image(pixels.data(), 2, 2, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "reject unsupported interpolation");
}

} // namespace

int main() {
    test_bilinear_resize();
    test_normalization();
    test_torchvision_short_edge_geometry();
    test_bicubic_matches_whole_image_reference();
    test_bicubic_special_values();
    test_bicubic_weights_use_no_fused_multiply_add();
    test_invalid_interpolation();
    return failures;
}
