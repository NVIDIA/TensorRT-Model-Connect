/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov3/runtime/image_preprocess.h"

#include <algorithm>
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

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

// The previous per-element implementation, kept as the oracle for the
// restructured passes. Both accumulate each output element's taps from 0.0F in
// span order, so they agree bit for bit unless the compiler fuses the
// multiply-adds of one and not the other (about one ulp).
struct ReferenceSpan {
    int32_t first{0};
    std::vector<float> weights;
};

std::vector<ReferenceSpan> reference_spans(int32_t input_size, int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<ReferenceSpan> spans(static_cast<std::size_t>(output_size));
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const int32_t first = std::max(0, static_cast<int32_t>(center - support + 0.5));
        const int32_t end = std::min(input_size, static_cast<int32_t>(center + support + 0.5));
        auto& span = spans[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weights.resize(static_cast<std::size_t>(end - first));
        double weight_sum = 0.0;
        for (int32_t input_index = first; input_index < end; ++input_index) {
            const double distance =
                std::abs((static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale);
            const double weight = distance < 1.0 ? 1.0 - distance : 0.0;
            span.weights[static_cast<std::size_t>(input_index - first)] =
                static_cast<float>(weight);
            weight_sum += weight;
        }
        for (float& weight : span.weights)
            weight = static_cast<float>(static_cast<double>(weight) / weight_sum);
    }
    return spans;
}

float reference_tap_sum(const float* input, int32_t stride, const ReferenceSpan& span) {
    float result = 0.0F;
    for (std::size_t i = 0; i < span.weights.size(); ++i)
        result += input[(span.first + static_cast<int32_t>(i)) * stride] * span.weights[i];
    return result;
}

std::vector<float> reference_preprocess(const std::vector<float>& input, int32_t input_h,
                                        int32_t input_w,
                                        const trtmc::Dinov3PreprocessConfig& config) {
    const int32_t output_h = config.input_image_h;
    const int32_t output_w = config.input_image_w;
    std::vector<float> horizontal = input;
    if (input_w != output_w) {
        const auto spans = reference_spans(input_w, output_w);
        horizontal.assign(static_cast<std::size_t>(input_h) * output_w * 3U, 0.0F);
        for (int32_t y = 0; y < input_h; ++y)
            for (int32_t x = 0; x < output_w; ++x)
                for (int32_t c = 0; c < 3; ++c)
                    horizontal[(static_cast<std::size_t>(y) * output_w + x) * 3U + c] =
                        reference_tap_sum(input.data() +
                                              static_cast<std::size_t>(y) * input_w * 3U + c,
                                          3, spans[static_cast<std::size_t>(x)]);
    }
    std::vector<float> resized = horizontal;
    if (input_h != output_h) {
        const auto spans = reference_spans(input_h, output_h);
        resized.assign(static_cast<std::size_t>(output_h) * output_w * 3U, 0.0F);
        for (int32_t y = 0; y < output_h; ++y)
            for (int32_t x = 0; x < output_w; ++x)
                for (int32_t c = 0; c < 3; ++c)
                    resized[(static_cast<std::size_t>(y) * output_w + x) * 3U + c] =
                        reference_tap_sum(horizontal.data() + static_cast<std::size_t>(x) * 3U + c,
                                          output_w * 3, spans[static_cast<std::size_t>(y)]);
    }
    const auto plane = static_cast<std::size_t>(output_h) * output_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
        for (std::size_t c = 0; c < 3; ++c)
            pixel_values[c * plane + pixel] =
                (resized[pixel * 3U + c] - config.image_mean[c]) / config.image_std[c];
    return pixel_values;
}

trtmc::Dinov3PreprocessConfig identity_config(int32_t height, int32_t width) {
    trtmc::Dinov3PreprocessConfig config;
    config.input_image_h = height;
    config.input_image_w = width;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    return config;
}

void test_defaults_match_hugging_face_processor() {
    const trtmc::Dinov3PreprocessConfig config;
    check(config.input_image_h == 224 && config.input_image_w == 224,
          "DINOv3 default processor size");
    check(config.image_mean == std::vector<float>({0.485F, 0.456F, 0.406F}),
          "DINOv3 ImageNet mean");
    check(config.image_std == std::vector<float>({0.229F, 0.224F, 0.225F}), "DINOv3 ImageNet std");
}

void test_direct_resize_preserves_rectangular_image_extent() {
    // A direct 1x2 -> 2x2 resize retains both columns. A shortest-edge resize
    // plus center crop would discard part of this red/blue extent.
    const std::vector<float> pixels = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 1, 2, identity_config(2, 2));
    const std::vector<float> expected = {
        1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
    };
    check(output == expected, "DINOv3 direct resize and NCHW layout");
}

void test_bilinear_resize_blends_center_pixel() {
    const std::vector<float> pixels = {
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 2, 2, identity_config(3, 3));
    check(output.size() == 27, "DINOv3 bilinear output size");
    if (output.size() == 27)
        check_close(output[4], 0.5F, 1e-6F, "DINOv3 bilinear center value");
}

void test_resize_matches_hugging_face_torchvision_golden() {
    // Golden from DINOv3ViTImageProcessor's torchvision backend:
    // rescale to float, direct bilinear resize to 3x3 with antialiasing, NCHW.
    const std::vector<float> pixels_u8 = {
        0,   10, 20,  64,  74, 84, 192, 202, 212, 255, 245, 235,
        255, 0,  128, 192, 32, 96, 64,  224, 48,  0,   255, 16,
    };
    std::vector<float> pixels = pixels_u8;
    for (float& value : pixels)
        value /= 255.0F;
    const std::vector<float> expected = {
        0.0752941221F, 0.5019608140F, 0.9258823991F, 0.5005882382F, 0.5019608140F, 0.5005882382F,
        0.9258823395F, 0.5019608140F, 0.0752940923F, 0.1145098135F, 0.5411764979F, 0.9101961255F,
        0.0760784373F, 0.5215686560F, 0.9368628263F, 0.0376470610F, 0.5019608140F, 0.9635294676F,
        0.1537255049F, 0.5803921819F, 0.8945097923F, 0.3090196252F, 0.4313725829F, 0.4974509776F,
        0.4643137455F, 0.2823529541F, 0.1003921479F,
    };
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 2, 4, identity_config(3, 3));
    check(output.size() == expected.size(), "DINOv3 HF resize golden size");
    if (output.size() != expected.size())
        return;
    for (std::size_t i = 0; i < output.size(); ++i)
        check_close(output[i], expected[i], 2e-7F, "DINOv3 HF resize golden value");
}

void test_imagenet_normalization_is_applied_after_resize() {
    const std::vector<float> pixels = {0.485F, 0.456F, 0.406F};
    trtmc::Dinov3PreprocessConfig config;
    config.input_image_h = 1;
    config.input_image_w = 1;
    const auto output = trtmc::preprocess_dinov3_image(pixels.data(), 1, 1, config);
    check(output.size() == 3, "DINOv3 normalized output size");
    for (float value : output)
        check_close(value, 0.0F, 1e-6F, "DINOv3 ImageNet normalization");
}

void test_matches_the_per_element_reference() {
    struct Case {
        int32_t input_h;
        int32_t input_w;
        int32_t output_h;
        int32_t output_w;
    };
    const Case cases[] = {
        {382, 640, 224, 224}, // both axes downscaled
        {2, 4, 3, 3},         // height upscaled, width downscaled
        {5, 7, 11, 13},       // both axes upscaled
        {224, 224, 224, 224}, // identity
        {100, 224, 224, 224}, // height only
        {224, 300, 224, 224}, // width only
        {17, 33, 7, 5},       // odd sizes
    };
    uint32_t state = 12345U;
    for (const auto& item : cases) {
        std::vector<float> pixels(static_cast<std::size_t>(item.input_h) * item.input_w * 3U);
        for (float& value : pixels) {
            state = state * 1664525U + 1013904223U;
            value = static_cast<float>(state >> 8) / 16777216.0F;
        }
        trtmc::Dinov3PreprocessConfig config;
        config.input_image_h = item.output_h;
        config.input_image_w = item.output_w;
        const auto actual =
            trtmc::preprocess_dinov3_image(pixels.data(), item.input_h, item.input_w, config);
        const auto expected = reference_preprocess(pixels, item.input_h, item.input_w, config);
        check(actual.size() == expected.size(), "DINOv3 reference comparison size");
        if (actual.size() != expected.size())
            continue;
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            // Written so that a NaN on either side counts as a mismatch.
            if (!(std::fabs(actual[i] - expected[i]) <= 1e-6F))
                ++mismatches;
        }
        check(mismatches == 0, "DINOv3 preprocessing matches the per-element reference");
    }
}

void test_invalid_input_is_rejected() {
    bool threw = false;
    try {
        (void)trtmc::preprocess_dinov3_image(nullptr, 0, 0, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DINOv3 empty input rejected");

    threw = false;
    try {
        const std::vector<float> pixels(3, 0.0F);
        auto config = identity_config(1, 1);
        config.image_std[1] = 0.0F;
        (void)trtmc::preprocess_dinov3_image(pixels.data(), 1, 1, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "DINOv3 zero std rejected");
}

} // namespace

int main() {
    test_defaults_match_hugging_face_processor();
    test_direct_resize_preserves_rectangular_image_extent();
    test_bilinear_resize_blends_center_pixel();
    test_resize_matches_hugging_face_torchvision_golden();
    test_imagenet_normalization_is_applied_after_resize();
    test_matches_the_per_element_reference();
    test_invalid_input_is_rejected();

    if (g_failures != 0) {
        std::cerr << g_failures << " DINOv3 preprocessing test(s) failed\n";
        return 1;
    }
    return 0;
}
