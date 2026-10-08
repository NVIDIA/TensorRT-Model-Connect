/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rt_detr_v2/runtime/box_decode.h"
#include "families/rt_detr_v2/runtime/image_preprocess_seam.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}

void test_preprocessing_rescales_without_normalising() {
    // Input arrives in [0, 1] from the CLI's decoder. A uniform mid-grey must
    // come back unchanged; dividing by 255 again would make it 0.002.
    // If the listed ImageNet statistics were applied it would be about 0.08
    // on the red channel and clearly outside [0, 1] on others.
    const int32_t h = 37, w = 53;
    std::vector<float> image(static_cast<std::size_t>(h) * w * 3, 128.0F / 255.0F);
    trtmc::RtDetrPreprocessConfig config;
    const auto out = trtmc::preprocess_rt_detr_image(image.data(), h, w, config);
    check(out.size() == static_cast<std::size_t>(3) * 640 * 640, "output is CHW 3x640x640");
    bool all_grey = true;
    for (const float value : out)
        all_grey = all_grey && std::fabs(value - 128.0F / 255.0F) < 1e-5F;
    check(all_grey, "a uniform image stays uniform at value/255");
}

void test_preprocessing_does_not_preserve_aspect_ratio() {
    // The reference stretches a 640x382 source to a 640x640 square.
    const int32_t h = 382, w = 640;
    std::vector<float> image(static_cast<std::size_t>(h) * w * 3, 0.0F);
    // Put a bright row at the bottom; after stretching it must still be at the
    // bottom, not letterboxed into the middle.
    for (int32_t column = 0; column < w; ++column)
        for (int32_t channel = 0; channel < 3; ++channel)
            image[(static_cast<std::size_t>(h - 1) * w + column) * 3 + channel] = 1.0F;
    const auto out =
        trtmc::preprocess_rt_detr_image(image.data(), h, w, trtmc::RtDetrPreprocessConfig{});
    const std::size_t plane = static_cast<std::size_t>(640) * 640;
    check(out[plane - 1] > 0.9F, "the last row stays bright after the stretch");
    check(out[plane / 2] < 0.1F, "the middle stays dark, so nothing was letterboxed");
}

void test_decode_matches_the_reference_detection() {
    // The reference's top detection for the shared test image: query box
    // (0.6764, 0.6140, 0.5136, 0.7085) in normalised cxcywh, class 2, on a
    // 640x382 source, giving xyxy about (268.56, 99.22, 597.27, 369.88).
    const int32_t queries = 1, classes = 3;
    std::vector<float> logits(static_cast<std::size_t>(queries) * classes, -9.0F);
    logits[2] = 2.8318F; // sigmoid(2.8318) is about 0.9444
    const std::vector<float> boxes{0.6764F, 0.6140F, 0.5136F, 0.7085F};
    const auto out =
        trtmc::decode_rt_detr_boxes(logits.data(), boxes.data(), queries, classes, 382, 640, 0.3F);
    check(out.size() == 1U, "one detection clears the threshold");
    if (out.empty())
        return;
    check(out[0].label == 2, "label comes from the flattened index modulo classes");
    check(std::fabs(out[0].score - 0.9444F) < 1e-3F, "score is the sigmoid of the logit");
    check(std::fabs(out[0].x_min - 268.56F) < 0.1F, "x_min matches the reference");
    check(std::fabs(out[0].y_min - 99.22F) < 0.1F, "y_min matches the reference");
    check(std::fabs(out[0].x_max - 597.27F) < 0.1F, "x_max matches the reference");
    check(std::fabs(out[0].y_max - 369.88F) < 0.1F, "y_max matches the reference");
}

void test_decode_can_emit_one_query_twice() {
    // The selection is top-k over the flattened query-by-class matrix, so a
    // query that scores highly for two classes appears twice. A per-query
    // argmax would return it once, and the two rules are indistinguishable on
    // any image where every winning query has a single dominant class.
    const int32_t queries = 2, classes = 2;
    const std::vector<float> logits{3.0F, 2.5F, -8.0F, -8.0F};
    const std::vector<float> boxes{0.5F, 0.5F, 0.2F, 0.2F, 0.1F, 0.1F, 0.05F, 0.05F};
    const auto out =
        trtmc::decode_rt_detr_boxes(logits.data(), boxes.data(), queries, classes, 100, 100, 0.5F);
    check(out.size() == 2U, "both classes of the same query are kept");
    if (out.size() < 2U)
        return;
    check(out[0].label == 0 && out[1].label == 1, "the two labels are distinct");
    check(std::fabs(out[0].x_min - out[1].x_min) < 1e-5F, "and they share one box");
}

} // namespace

int main() {
    test_preprocessing_rescales_without_normalising();
    test_preprocessing_does_not_preserve_aspect_ratio();
    test_decode_matches_the_reference_detection();
    test_decode_can_emit_one_query_twice();
    if (failures != 0) {
        std::fprintf(stderr, "%d rt_detr_v2 check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("rt_detr_v2 preprocessing and decoding checks passed\n");
    return EXIT_SUCCESS;
}
