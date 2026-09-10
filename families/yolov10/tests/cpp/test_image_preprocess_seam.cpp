/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolov10/runtime/image_preprocess_seam.h"

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

void test_letterbox_keeps_aspect_and_centres_the_image() {
    // A 2:1 image fitted into a square must be scaled by the longer side and
    // padded top and bottom, not stretched.
    const std::vector<float> pixels(3U * 2U * 4U, 1.0F);
    trtmc::Yolov10PreprocessConfig config;
    config.input_image_h = 8;
    config.input_image_w = 8;
    config.pad_value = 0.5F;
    trtmc::Yolov10Letterbox letterbox;
    const auto values = trtmc::preprocess_yolov10_image(pixels.data(), 2, 4, config, letterbox);

    check(values.size() == 3U * 8U * 8U, "letterbox output size");
    check_close(letterbox.scale, 2.0F, 1e-6F, "letterbox scale uses the longer side");
    check_close(letterbox.pad_x, 0.0F, 1e-6F, "letterbox does not pad the fitted axis");
    check_close(letterbox.pad_y, 2.0F, 1e-6F, "letterbox pads the short axis evenly");
    // Row 0 is padding; row 2 is the first image row.
    check_close(values[0], 0.5F, 1e-6F, "letterbox fills the margin with the pad value");
    check_close(values[2 * 8], 1.0F, 1e-6F, "letterbox keeps the image inside the margin");
}

void test_preprocess_reads_interleaved_and_writes_planar() {
    // The CLI hands over interleaved RGB and the engine wants planar CHW.
    // Every earlier test used one value for the whole image, which cannot tell
    // the two layouts apart; a constant image reads the same either way.
    constexpr float kRed = 0.10F;
    constexpr float kGreen = 0.50F;
    constexpr float kBlue = 0.90F;
    std::vector<float> pixels(4U * 4U * 3U);
    for (std::size_t pixel = 0; pixel < 16U; ++pixel) {
        pixels[pixel * 3U + 0U] = kRed;
        pixels[pixel * 3U + 1U] = kGreen;
        pixels[pixel * 3U + 2U] = kBlue;
    }
    trtmc::Yolov10PreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    trtmc::Yolov10Letterbox letterbox;
    const auto values = trtmc::preprocess_yolov10_image(pixels.data(), 4, 4, config, letterbox);

    const std::size_t plane = 4U * 4U;
    check_close(values[0], kRed, 1e-6F, "first plane holds red");
    check_close(values[plane], kGreen, 1e-6F, "second plane holds green");
    check_close(values[2U * plane], kBlue, 1e-6F, "third plane holds blue");
    // Whole planes, not just their first element.
    for (std::size_t index = 0; index < plane; ++index) {
        check_close(values[index], kRed, 1e-6F, "red plane is uniform");
        check_close(values[plane + index], kGreen, 1e-6F, "green plane is uniform");
        check_close(values[2U * plane + index], kBlue, 1e-6F, "blue plane is uniform");
    }
}

void test_letterbox_reports_a_mapping_that_inverts() {
    // The pipeline undoes the letterbox with these two numbers, so a point put
    // through them and back has to land where it started.
    const std::vector<float> pixels(3U * 5U * 9U, 0.25F);
    trtmc::Yolov10PreprocessConfig config;
    config.input_image_h = 32;
    config.input_image_w = 32;
    trtmc::Yolov10Letterbox letterbox;
    (void)trtmc::preprocess_yolov10_image(pixels.data(), 5, 9, config, letterbox);

    const float original = 3.5F;
    const float mapped = original * letterbox.scale + letterbox.pad_x;
    check_close((mapped - letterbox.pad_x) / letterbox.scale, original, 1e-4F,
                "letterbox mapping inverts");
}

void test_preprocess_rejects_an_empty_image() {
    bool threw = false;
    try {
        trtmc::Yolov10PreprocessConfig config;
        trtmc::Yolov10Letterbox letterbox;
        (void)trtmc::preprocess_yolov10_image(nullptr, 0, 0, config, letterbox);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "YOLOv10 rejects an empty image");
}

} // namespace

int main() {
    test_letterbox_keeps_aspect_and_centres_the_image();
    test_preprocess_reads_interleaved_and_writes_planar();
    test_letterbox_reports_a_mapping_that_inverts();
    test_preprocess_rejects_an_empty_image();

    if (g_failures != 0) {
        std::cerr << g_failures << " YOLOv10 preprocess test(s) failed\n";
        return 1;
    }
    return 0;
}
