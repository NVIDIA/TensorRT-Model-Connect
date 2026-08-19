/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// test_qwen_vl_sampler.cpp -- CPU tests for Qwen-VL EOS sampling behavior
// =============================================================================

#include "sampler.h"
#include "trtmc/pipeline.h"

#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtmc::QwenVlSampleResult greedy_sample(const trtmc::QwenVlSamplingParams& params,
                                               int32_t winning_token) {
    std::vector<float> logits(8, 0.0F);
    logits[static_cast<std::size_t>(winning_token)] = 1.0F;
    auto sampler = trtmc::create_qwen_vl_sampler(params);
    return sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
}

static void test_any_default_eos_stops_generation() {
    trtmc::GenerateConfig config;
    const std::vector<int32_t> defaults{5, 7};
    const auto params = trtmc::qwen_vl_sampling_params_from_config(config, defaults);

    check(params.eos_token_ids == defaults, "sampler: preserves all default EOS IDs");
    check(greedy_sample(params, 7).is_eos, "sampler: second default EOS stops generation");
}

static void test_request_eos_overrides_model_defaults() {
    trtmc::GenerateConfig config;
    config.eos_token_id = 3;
    const auto params =
        trtmc::qwen_vl_sampling_params_from_config(config, std::vector<int32_t>{5, 7});

    check(params.eos_token_ids == std::vector<int32_t>({3}),
          "sampler: request EOS replaces model defaults");
    check(!greedy_sample(params, 7).is_eos,
          "sampler: overridden model EOS no longer stops generation");
    check(greedy_sample(params, 3).is_eos, "sampler: explicit request EOS stops generation");
}

int main() {
    test_any_default_eos_stops_generation();
    test_request_eos_overrides_model_defaults();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All Qwen-VL sampler tests passed.\n";
    return 0;
}
