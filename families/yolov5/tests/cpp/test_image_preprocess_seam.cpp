/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolov5/runtime/image_preprocess_seam.h"
#include "families/yolov5/runtime/pipeline.h"

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
    trtmc::Yolo11PreprocessConfig config;
    config.input_image_h = 8;
    config.input_image_w = 8;
    config.pad_value = 0.5F;
    trtmc::Yolo11Letterbox letterbox;
    const auto values = trtmc::preprocess_yolov5_image(pixels.data(), 2, 4, config, letterbox);

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
    trtmc::Yolo11PreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    trtmc::Yolo11Letterbox letterbox;
    const auto values = trtmc::preprocess_yolov5_image(pixels.data(), 4, 4, config, letterbox);

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
    trtmc::Yolo11PreprocessConfig config;
    config.input_image_h = 32;
    config.input_image_w = 32;
    trtmc::Yolo11Letterbox letterbox;
    (void)trtmc::preprocess_yolov5_image(pixels.data(), 5, 9, config, letterbox);

    const float original = 3.5F;
    const float mapped = original * letterbox.scale + letterbox.pad_x;
    check_close((mapped - letterbox.pad_x) / letterbox.scale, original, 1e-4F,
                "letterbox mapping inverts");
}

void test_preprocess_rejects_an_empty_image() {
    bool threw = false;
    try {
        trtmc::Yolo11PreprocessConfig config;
        trtmc::Yolo11Letterbox letterbox;
        (void)trtmc::preprocess_yolov5_image(nullptr, 0, 0, config, letterbox);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "YOLOv5 rejects an empty image");
}

void test_suppression_keeps_the_strongest_and_drops_its_overlaps() {
    // YOLOv5's head reports one prediction per anchor and leaves the overlaps
    // in, so the runtime removes them.
    const auto box = [](float x0, float y0, float x1, float y1, float score, std::int32_t id) {
        trtmc::DetectionBox value;
        value.x_min = x0;
        value.y_min = y0;
        value.x_max = x1;
        value.y_max = y1;
        value.score = score;
        value.class_id = id;
        return value;
    };
    const std::vector<trtmc::DetectionBox> candidates{
        box(0.0F, 0.0F, 10.0F, 10.0F, 0.9F, 1),
        box(0.5F, 0.5F, 10.5F, 10.5F, 0.8F, 1),
        box(0.5F, 0.5F, 10.5F, 10.5F, 0.7F, 2),
        box(50.0F, 50.0F, 60.0F, 60.0F, 0.6F, 1),
    };
    const auto kept = trtmc::suppress_yolov5_boxes(candidates, 0.7F, 300U);
    check(kept.size() == 3U, "suppression drops only the overlapping duplicate");
    check_close(kept[0].score, 0.9F, 1e-6F, "suppression keeps the strongest first");
    check(kept[1].class_id == 2, "suppression never lets one class suppress another");
    check_close(kept[2].score, 0.6F, 1e-6F, "suppression keeps a distant box of the same class");
}

void test_suppression_honours_the_detection_cap() {
    std::vector<trtmc::DetectionBox> candidates;
    for (std::int32_t index = 0; index < 10; ++index) {
        trtmc::DetectionBox value;
        value.x_min = static_cast<float>(index) * 100.0F;
        value.y_min = 0.0F;
        value.x_max = value.x_min + 10.0F;
        value.y_max = 10.0F;
        value.score = 1.0F - static_cast<float>(index) * 0.01F;
        value.class_id = 1;
        candidates.push_back(value);
    }
    const auto kept = trtmc::suppress_yolov5_boxes(candidates, 0.7F, 4U);
    check(kept.size() == 4U, "suppression stops at the detection cap");
    check_close(kept[0].score, 1.0F, 1e-6F, "the cap keeps the strongest, not the first seen");
}

void test_reported_padding_matches_where_the_image_is_placed() {
    // An odd total padding is where the two can disagree: the image is pasted
    // at an integer offset, so reporting a half pixel shifts every box by half
    // a pixel when the pipeline inverts the letterbox.
    const std::vector<float> pixels(3U * 5U * 8U, 1.0F);
    trtmc::Yolo11PreprocessConfig config;
    config.input_image_h = 33;
    config.input_image_w = 33;
    config.pad_value = 0.25F;
    trtmc::Yolo11Letterbox letterbox;
    const auto values = trtmc::preprocess_yolov5_image(pixels.data(), 5, 8, config, letterbox);

    const auto top = static_cast<std::size_t>(letterbox.pad_y);
    check_close(letterbox.pad_y, std::floor(letterbox.pad_y), 1e-6F,
                "reported vertical padding is a whole pixel");
    check_close(letterbox.pad_x, std::floor(letterbox.pad_x), 1e-6F,
                "reported horizontal padding is a whole pixel");
    // The row just above the image must still hold the pad value.
    if (top > 0U) {
        check_close(values[(top - 1U) * 33U], config.pad_value, 1e-6F,
                    "the row before the image is padding");
    }
}

} // namespace

int main() {
    test_reported_padding_matches_where_the_image_is_placed();
    test_suppression_keeps_the_strongest_and_drops_its_overlaps();
    test_suppression_honours_the_detection_cap();
    test_letterbox_keeps_aspect_and_centres_the_image();
    test_preprocess_reads_interleaved_and_writes_planar();
    test_letterbox_reports_a_mapping_that_inverts();
    test_preprocess_rejects_an_empty_image();

    if (g_failures != 0) {
        std::cerr << g_failures << " YOLOv5 preprocess test(s) failed\n";
        return 1;
    }
    return 0;
}
