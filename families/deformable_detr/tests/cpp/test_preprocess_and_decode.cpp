/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deformable_detr/runtime/box_decode.h"
#include "families/deformable_detr/runtime/image_preprocess_seam.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

std::vector<float> flat_image(std::int32_t height, std::int32_t width, float value) {
    return std::vector<float>(static_cast<std::size_t>(height) * width * 3, value);
}

void test_preprocess_normalises_with_the_configured_statistics() {
    trtmc::DeformableDetrPreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    const auto pixels = flat_image(8, 8, 0.485F);
    const auto out = trtmc::preprocess_deformable_detr_image(pixels.data(), 8, 8, config);
    check(out.size() == 3U * 4U * 4U, "the output is CHW at the configured size");
    // Channel 0's mean is exactly the input, so it normalises to ~0; the other
    // two channels have different means and must not.
    check(std::fabs(out[0]) < 2e-2F, "the red channel centres on zero");
    check(std::fabs(out[16]) > 1e-2F, "the green channel is not centred");
}

void test_preprocess_can_skip_normalisation() {
    trtmc::DeformableDetrPreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    config.do_normalize = false;
    const auto pixels = flat_image(2, 2, 0.5F);
    const auto out = trtmc::preprocess_deformable_detr_image(pixels.data(), 2, 2, config);
    // The input is already in [0, 1]; without normalisation it survives the
    // 8-bit round trip and nothing else touches it.
    for (const auto value : out)
        check(std::fabs(value - 0.5F) < 1.0F / 255.0F, "the pixel passes through unscaled");
}

void test_preprocess_rejects_an_empty_image() {
    trtmc::DeformableDetrPreprocessConfig config;
    bool threw = false;
    try {
        (void)trtmc::preprocess_deformable_detr_image(nullptr, 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is rejected");
}

void test_decode_selects_across_classes_not_per_query() {
    // Two queries, three classes. Query 0 is strong in two classes at once;
    // a per-query argmax would emit it once, top-k emits it twice.
    const std::vector<float> logits{3.0F, 2.5F, -5.0F, -5.0F, -5.0F, -5.0F};
    const std::vector<float> boxes{0.5F, 0.5F, 0.2F, 0.2F, 0.1F, 0.1F, 0.1F, 0.1F};
    const auto out =
        trtmc::decode_deformable_detr_boxes(logits.data(), boxes.data(), 2, 3, 100, 200, 0.5F, 100);
    check(out.size() == 2U, "both classes of the same query survive");
    check(out[0].label == 0 && out[1].label == 1, "labels come from the class index");
    check(out[0].score > out[1].score, "detections are ranked by score");
}

void test_decode_scales_boxes_by_the_original_size() {
    const std::vector<float> logits{4.0F};
    // cxcywh = (0.5, 0.5, 0.5, 0.5) over a 200 wide, 100 tall image.
    const std::vector<float> boxes{0.5F, 0.5F, 0.5F, 0.5F};
    const auto out =
        trtmc::decode_deformable_detr_boxes(logits.data(), boxes.data(), 1, 1, 100, 200, 0.1F, 100);
    check(out.size() == 1U, "the detection survives the threshold");
    check(std::fabs(out[0].x_min - 50.0F) < 1e-4F, "x scales by the width");
    check(std::fabs(out[0].x_max - 150.0F) < 1e-4F, "x scales by the width");
    check(std::fabs(out[0].y_min - 25.0F) < 1e-4F, "y scales by the height");
    check(std::fabs(out[0].y_max - 75.0F) < 1e-4F, "y scales by the height");
}

void test_decode_honours_top_k_rather_than_the_query_count() {
    // Six query-class slots all above threshold, but top_k keeps two. Passing
    // the query count instead - as RT-DETR does - would keep all six.
    const std::vector<float> logits{5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.5F};
    const std::vector<float> boxes{0.5F, 0.5F, 0.2F, 0.2F, 0.5F, 0.5F, 0.2F, 0.2F};
    const auto out =
        trtmc::decode_deformable_detr_boxes(logits.data(), boxes.data(), 2, 3, 10, 10, 0.0F, 2);
    check(out.size() == 2U, "top_k bounds the detection count");
}

void test_decode_rejects_a_non_positive_top_k() {
    const std::vector<float> logits{1.0F};
    const std::vector<float> boxes{0.5F, 0.5F, 0.1F, 0.1F};
    bool threw = false;
    try {
        (void)trtmc::decode_deformable_detr_boxes(logits.data(), boxes.data(), 1, 1, 8, 8, 0.0F, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a non-positive top_k is rejected");
}

} // namespace

int main() {
    test_preprocess_normalises_with_the_configured_statistics();
    test_preprocess_can_skip_normalisation();
    test_preprocess_rejects_an_empty_image();
    test_decode_selects_across_classes_not_per_query();
    test_decode_scales_boxes_by_the_original_size();
    test_decode_honours_top_k_rather_than_the_query_count();
    test_decode_rejects_a_non_positive_top_k();
    if (failures != 0) {
        std::fprintf(stderr, "%d deformable_detr check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("deformable_detr preprocess and decode checks passed\n");
    return EXIT_SUCCESS;
}
