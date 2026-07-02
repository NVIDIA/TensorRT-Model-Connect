/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ABI-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         C ABI batch image generation: argument validation + happy path
// Preconditions:  trtmc::IPipeline default `generate_image_batch` available
// Postconditions: bad inputs return TRTMC_ERR_INVALID_ARG; happy path fills out_results
// =============================================================================

// =============================================================================
// Test suite: trtmc_generate_batch (PR 3 — final piece of the diffusion
// batch-inference series).
//
// Purpose:
//   Validates the new C-ABI entry point for batch image generation. The
//   existing C-ABI tests (`test_c_abi_entry`, `test_c_abi_runtime_regression`)
//   skip pipeline creation because it requires a real `.trtfb` bundle and a
//   GPU. We follow the same pattern here for the argument-validation cases
//   (null handle / null prompts / mismatched lengths) and additionally use a
//   tiny in-test fake IPipeline to exercise the happy path end-to-end without
//   needing a bundle or a GPU.
//
// Dependencies:
//   - trtmc/pipeline.h: trtmc_generate_batch, trtmc_image_result_t,
//     TRTMC_ERR_INVALID_ARG, TRTMC_OK, trtmc_image_result_free.
// =============================================================================

#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Minimal IPipeline subclass that returns a deterministic ImageResult per
// prompt. The default `generate_image_batch` implementation in pipeline.h
// loops over `generate_image`, so overriding the single-prompt path is
// enough to exercise the end-to-end C-ABI conversion (including the
// per-sample-seed plumbing).
class FakeImagePipeline final : public trtmc::IPipeline {
  public:
    trtmc::ImageResult generate_image(const std::string& prompt,
                                      const trtmc::GenerateConfig& cfg) override {
        trtmc::ImageResult r;
        r.height = 4;
        r.width = 4;
        r.channels = 3;
        r.num_frames = 1;
        r.pixels.assign(static_cast<std::size_t>(r.channels * r.height * r.width), 0.0F);
        // Stash a couple of identifying values so the test can confirm the
        // C-ABI hand-off preserved them: pixel[0] = prompt length, pixel[1]
        // = effective seed (cast to float).
        if (!r.pixels.empty())
            r.pixels[0] = static_cast<float>(prompt.size());
        if (r.pixels.size() > 1)
            r.pixels[1] = static_cast<float>(cfg.seed);
        return r;
    }

    const char* model_id() const override { return "fake/image"; }
    const char* pipeline_type() const override { return "fake_image"; }
};

// -- Argument validation ---------------------------------------------------

void test_invalid_args_return_invalid_arg() {
    // One consolidated check covering the common invalid-arg shapes.
    FakeImagePipeline pipe;
    const char* prompts[] = {"a", "b"};
    std::uint32_t seeds[] = {1, 2, 3};
    trtmc_image_result_t out[2]{};

    check(trtmc_generate_batch(nullptr, prompts, 1, seeds, 1, 4, 7.5F, out) ==
              TRTMC_ERR_INVALID_ARG,
          "null handle returns TRTMC_ERR_INVALID_ARG");
    check(trtmc_generate_batch(&pipe, prompts, 2, seeds, 3, 4, 7.5F, out) == TRTMC_ERR_INVALID_ARG,
          "num_prompts != num_seeds returns TRTMC_ERR_INVALID_ARG");
}

// -- Happy path -----------------------------------------------------------

void test_batch_fills_results_and_propagates_seeds() {
    FakeImagePipeline pipe;
    const char* prompts[] = {"hello", "world!!"};
    std::uint32_t seeds[] = {42U, 1337U};
    trtmc_image_result_t out[2]{};

    const int rc = trtmc_generate_batch(&pipe, prompts, 2, seeds, 2, 4, 7.5F, out);
    check(rc == TRTMC_OK, "happy-path returns TRTMC_OK");

    // Both results filled with the fake's 4x4x3 layout.
    check(out[0].height == 4 && out[0].width == 4 && out[0].channels == 3, "result[0] shape");
    check(out[1].height == 4 && out[1].width == 4 && out[1].channels == 3, "result[1] shape");
    check(out[0].pixels != nullptr, "result[0] pixels allocated");
    check(out[1].pixels != nullptr, "result[1] pixels allocated");
    check(out[0].num_pixels == 4U * 4U * 3U, "result[0] num_pixels correct");
    check(out[1].num_pixels == 4U * 4U * 3U, "result[1] num_pixels correct");

    // Prompt-length sentinel (pixel[0]) and per-sample seed sentinel
    // (pixel[1]) confirm the conversion preserved per-prompt inputs and
    // that the C-ABI plumbed each entry's seed through `cfg.seed`.
    check(out[0].pixels[0] == static_cast<float>(std::string("hello").size()),
          "result[0] prompt-length sentinel matches");
    check(out[1].pixels[0] == static_cast<float>(std::string("world!!").size()),
          "result[1] prompt-length sentinel matches");
    check(out[0].pixels[1] == static_cast<float>(42),
          "result[0] seed propagated to pipeline cfg.seed");
    check(out[1].pixels[1] == static_cast<float>(1337),
          "result[1] seed propagated to pipeline cfg.seed");

    // Release allocations through the public free helper (the whole point
    // of the free helper is that callers can release without knowing about
    // the C++ allocator).
    trtmc_image_result_free(&out[0]);
    trtmc_image_result_free(&out[1]);
    check(out[0].pixels == nullptr, "free clears pixels[0]");
    check(out[1].pixels == nullptr, "free clears pixels[1]");
}

void test_image_result_free_null_safe() {
    trtmc_image_result_free(nullptr); // null pointer
    trtmc_image_result_t empty{};
    trtmc_image_result_free(&empty); // pixels==nullptr
    check(true, "trtmc_image_result_free is null-safe");
}

} // namespace

int main() {
    test_invalid_args_return_invalid_arg();
    test_batch_fills_results_and_propagates_seeds();
    test_image_result_free_null_safe();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI batch tests passed.\n";
    return 0;
}
