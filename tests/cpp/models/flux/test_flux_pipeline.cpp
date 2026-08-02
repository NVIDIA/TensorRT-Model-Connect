/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/flux/flux_clip_helpers.h"
#include "runtime/models/flux/pipeline.h"

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

void test_flux_construction() {
    trtmc::FluxDiffusionConfig cfg;
    trtmc::FluxPreprocessorWeights weights;

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, weights, nullptr, nullptr, "test-flux");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline", "FluxPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-flux", "FluxPipeline model_id");
}

void test_flux_with_custom_config() {
    trtmc::FluxDiffusionConfig cfg;
    cfg.video_height = 256;
    cfg.video_width = 256;
    cfg.scale_factor_spatial = 8;
    cfg.patch_size = {1, 2, 2};

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, trtmc::FluxPreprocessorWeights{},
                                 nullptr, nullptr, "test-flux-custom");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline",
          "FluxPipeline custom config pipeline_type");
}

void test_clip_padding_preserves_eos_when_truncated() {
    using trtmc::diffusion::flux_clip::pad_and_truncate_ids;

    check(pad_and_truncate_ids({10, 1, 11}, 5, 11, 11) == std::vector<int32_t>({10, 1, 11, 11, 11}),
          "CLIP short input pads with EOS");
    check(pad_and_truncate_ids({10, 1, 2, 3, 11}, 5, 11, 11) ==
              std::vector<int32_t>({10, 1, 2, 3, 11}),
          "CLIP exact input preserves EOS");
    check(pad_and_truncate_ids({10, 1, 2, 3, 4, 11}, 5, 11, 11) ==
              std::vector<int32_t>({10, 1, 2, 3, 11}),
          "CLIP truncated input restores EOS");
}

} // namespace

int main() {
    test_flux_construction();
    test_flux_with_custom_config();
    test_clip_padding_preserves_eos_when_truncated();
    if (failures > 0) {
        std::cerr << failures << " flux pipeline test(s) FAILED\n";
    }
    return failures;
}
