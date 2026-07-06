/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/z_image/pipeline.h"

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

void test_zimage_construction() {
    trtmc::ZImageDiffusionConfig cfg;
    trtmc::ZImageCommonPreprocessorWeights weights;
    trtmc::ZImagePreprocessorWeights z_weights;

    trtmc::ZImagePipeline pipeline(nullptr, nullptr, nullptr, cfg, weights, z_weights, nullptr,
                                   "test-zimage", "/tmp/test.trtfb");

    check(std::string(pipeline.pipeline_type()) == "ZImagePipeline",
          "ZImagePipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-zimage", "ZImagePipeline model_id");
}

void test_zimage_initial_latent_contract() {
    std::string error;
    const std::vector<float> supplied(16, 1.0F);
    check(trtmc::validate_z_image_initial_latents(supplied.size(), 1, supplied, error),
          "Z-Image accepts exact caller latent size");
    check(!trtmc::validate_z_image_initial_latents(supplied.size() + 1, 1, supplied, error),
          "Z-Image rejects wrong caller latent size");
    check(!trtmc::validate_z_image_initial_latents(supplied.size(), 2, supplied, error),
          "Z-Image rejects one latent for multiple prompts");
}

} // namespace

int main() {
    test_zimage_construction();
    test_zimage_initial_latent_contract();
    if (failures > 0) {
        std::cerr << failures << " z-image pipeline test(s) FAILED\n";
    }
    return failures;
}
