/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/z_image/gpu_matmul.h"
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

void test_zimage_text_encoder_input_shape() {
    check(trtmc::z_image_text_encoder_input_shape(1, 512) == std::vector<int64_t>({512}),
          "Z-Image static text encoder keeps rank-1 input");
    check(trtmc::z_image_text_encoder_input_shape(2, 512) == std::vector<int64_t>({1, 512}),
          "Z-Image dynamic text encoder adds batch dimension");
}

void test_zimage_attention_mask() {
    const auto mask = trtmc::make_z_image_attention_mask(2, 5, 3);
    check(mask.size() == 7, "Z-Image attention mask covers image and caption tokens");
    check(mask[0] == 0.0F && mask[1] == 0.0F && mask[2] == 0.0F && mask[3] == 0.0F &&
              mask[4] == 0.0F,
          "Z-Image attention mask keeps image and padded HF caption tokens visible");
    check(mask[5] < -1.0e8F && mask[6] < -1.0e8F,
          "Z-Image attention mask hides unused fixed caption slots");
}

void test_zimage_gpu_matmul_policy() {
    check(trtmc::z_image_should_use_gpu_matmul(4096, 64, 3840),
          "Z-Image 1024px patch embedding uses GPU matmul");
    check(trtmc::z_image_should_use_gpu_matmul(512, 2560, 3840),
          "Z-Image caption projection uses GPU matmul");
    check(!trtmc::z_image_should_use_gpu_matmul(1, 256, 3840),
          "Z-Image timestep projection stays on CPU below threshold");
    check(!trtmc::z_image_should_use_gpu_matmul(1, 16, 16), "Z-Image tiny projections stay on CPU");
}

} // namespace

int main() {
    test_zimage_construction();
    test_zimage_initial_latent_contract();
    test_zimage_text_encoder_input_shape();
    test_zimage_attention_mask();
    test_zimage_gpu_matmul_policy();
    if (failures > 0) {
        std::cerr << failures << " z-image pipeline test(s) FAILED\n";
    }
    return failures;
}
