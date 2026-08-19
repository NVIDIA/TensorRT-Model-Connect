/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"

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
    trtmc::WanDiffusionConfig cfg;
    trtmc::WanPreprocessorWeights weights;

    trtmc::WanPipeline pipeline(nullptr, nullptr, nullptr, cfg, weights, nullptr, "test-wan");

    check(std::string(pipeline.pipeline_type()) == "WanPipeline", "WanPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-wan", "WanPipeline model_id");
}

void test_wan_generation_failure_does_not_report_a_fake_frame() {
    trtmc::WanDiffusionConfig cfg;
    cfg.video_height = 384;
    cfg.video_width = 672;
    cfg.video_num_frames = 5;
    cfg.scale_factor_temporal = 4;
    cfg.scale_factor_spatial = 8;
    cfg.z_dim = 16;
    cfg.dit_dim = 1536;
    cfg.text_seq_len = 226;
    cfg.patch_size = {1, 2, 2};
    trtmc::WanPreprocessorWeights invalid_weights;
    trtmc::WanPipeline pipeline(nullptr, nullptr, nullptr, cfg, invalid_weights, nullptr,
                                "test-wan");

    const auto result = pipeline.generate_image("prompt");

    check(result.num_frames == 0, "Wan generation failure reports zero frames");
    check(result.pixels.empty(), "Wan generation failure has no pixel buffer");
}

} // namespace

int main() {
    test_wan_construction();
    test_wan_generation_failure_does_not_report_a_fake_frame();
    if (failures > 0) {
        std::cerr << failures << " wan pipeline test(s) FAILED\n";
    }
    return failures;
}
