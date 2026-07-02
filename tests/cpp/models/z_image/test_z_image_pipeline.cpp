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

} // namespace

int main() {
    test_zimage_construction();
    if (failures > 0) {
        std::cerr << failures << " z-image pipeline test(s) FAILED\n";
    }
    return failures;
}
