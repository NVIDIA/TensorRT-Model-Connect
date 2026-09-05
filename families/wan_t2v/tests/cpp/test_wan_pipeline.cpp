/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan_t2v/runtime/pipeline.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_wan_construction() {
    trtmc::WanDiffusionConfig config;
    trtmc::WanPreprocessorWeights weights;
    trtmc::WanPipeline pipeline(nullptr, nullptr, nullptr, config, weights, nullptr, "test-wan");
    check(std::string(pipeline.task()) == trtmc::IImageGeneration::kTask,
          "WanPipeline exposes the image-generation task");
}

void test_wan_generation_failure_does_not_report_a_fake_frame() {
    trtmc::WanDiffusionConfig config;
    config.video_height = 384;
    config.video_width = 672;
    config.video_num_frames = 5;
    config.scale_factor_temporal = 4;
    config.scale_factor_spatial = 8;
    config.z_dim = 16;
    config.dit_dim = 1536;
    config.text_seq_len = 226;
    config.patch_size = {1, 2, 2};
    trtmc::WanPreprocessorWeights invalid_weights;
    trtmc::WanPipeline pipeline(nullptr, nullptr, nullptr, config, invalid_weights, nullptr,
                                "test-wan");

    const auto result = pipeline.generate_image("prompt");

    check(result.num_frames == 0, "Wan generation failure reports zero frames");
    check(result.pixels.empty(), "Wan generation failure has no pixel buffer");
}

} // namespace

int main() {
    test_wan_construction();
    test_wan_generation_failure_does_not_report_a_fake_frame();
    return failures == 0 ? 0 : 1;
}
