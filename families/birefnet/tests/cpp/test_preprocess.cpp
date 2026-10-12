/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/birefnet/runtime/image_preprocess_seam.h"

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

void test_preprocessing_normalises_with_imagenet_statistics() {
    // Input arrives in [0, 1] from the CLI's decoder. A uniform mid-grey must
    // come out as (0.5019 - mean) / std per channel, which is clearly outside
    // [0, 1] - the opposite of rt_detr_v2, which does not normalise at all.
    const int32_t h = 21, w = 33;
    std::vector<float> image(static_cast<std::size_t>(h) * w * 3, 128.0F / 255.0F);
    trtmc::BiRefNetPreprocessConfig config;
    const auto out = trtmc::preprocess_birefnet_image(image.data(), h, w, config);
    check(out.size() == static_cast<std::size_t>(3) * 1024 * 1024, "output is 3x1024x1024");
    const std::size_t plane = static_cast<std::size_t>(1024) * 1024;
    for (int32_t channel = 0; channel < 3; ++channel) {
        const float expected =
            (128.0F / 255.0F - config.image_mean[channel]) / config.image_std[channel];
        check(std::fabs(out[static_cast<std::size_t>(channel) * plane] - expected) < 1e-4F,
              "each channel uses its own mean and standard deviation");
    }
}

void test_mask_thresholds_and_resamples_to_the_source_size() {
    // A 2x2 logit map, positive only in the bottom right, widened to 4x4.
    const std::vector<float> logits{-8.0F, -8.0F, -8.0F, 8.0F};
    const auto mask = trtmc::mask_from_logits(logits.data(), 2, 2, 4, 4, 0.5F);
    check(mask.size() == 16U, "the mask comes back at the requested size");
    check(mask[0] == 0 && mask[15] == 1, "the positive corner survives the resample");
    int ones = 0;
    for (const int32_t value : mask)
        ones += value;
    check(ones == 4, "exactly one quadrant is foreground");
    for (const int32_t value : mask)
        check(value == 0 || value == 1, "only class ids 0 and 1 appear");
}

void test_mask_threshold_is_respected() {
    // sigmoid(0.5) is about 0.622, so a threshold above it must reject.
    const std::vector<float> logits{0.5F};
    check(trtmc::mask_from_logits(logits.data(), 1, 1, 1, 1, 0.5F)[0] == 1,
          "a logit of 0.5 clears a threshold of 0.5");
    check(trtmc::mask_from_logits(logits.data(), 1, 1, 1, 1, 0.8F)[0] == 0,
          "and fails a threshold of 0.8");
}

} // namespace

int main() {
    test_preprocessing_normalises_with_imagenet_statistics();
    test_mask_thresholds_and_resamples_to_the_source_size();
    test_mask_threshold_is_respected();
    if (failures != 0) {
        std::fprintf(stderr, "%d birefnet check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("birefnet preprocessing and mask checks passed\n");
    return EXIT_SUCCESS;
}
