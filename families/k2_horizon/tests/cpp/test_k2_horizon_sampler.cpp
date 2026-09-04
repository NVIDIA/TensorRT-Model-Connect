/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon/runtime/pipeline.h"
#include "families/k2_horizon/runtime/sampler.h"
#include "trtmc/task.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_greedy_ties_choose_the_lowest_index() {
    trtmc::TextGenerationConfig config;
    const auto params = trtmc::k2_horizon_sampling_params_from_config(config, 7);
    auto sampler = trtmc::create_k2_horizon_sampler(params);
    const std::vector<float> logits{0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F};
    const auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);

    check(result.token_id == 1, "greedy tie selects lowest vocabulary index");
    check(!result.is_eos, "non-EOS greedy token does not stop generation");
}

void test_all_model_eos_ids_are_preserved() {
    trtmc::TextGenerationConfig config;
    const std::vector<int32_t> defaults{5, 7};
    const auto params = trtmc::k2_horizon_sampling_params_from_config(config, defaults);
    auto sampler = trtmc::create_k2_horizon_sampler(params);
    std::vector<float> logits(8, 0.0F);
    logits[7] = 1.0F;

    check(sampler->sample(logits.data(), 8, params).is_eos, "secondary model EOS stops generation");
}

void test_non_greedy_requests_fail_closed() {
    trtmc::TextGenerationConfig config;
    config.top_k = 2;
    const auto params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    bool rejected = false;
    try {
        (void)trtmc::create_k2_horizon_sampler(params);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "non-greedy request is rejected");
}

bool sampling_params_are_rejected(const trtmc::K2HorizonSamplingParams& params,
                                  int32_t vocab_size = 8) {
    try {
        trtmc::k2_horizon_validate_sampling_params(params, vocab_size);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_sampling_controls_are_validated_before_greedy_decode() {
    trtmc::TextGenerationConfig config;
    auto params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    check(!sampling_params_are_rejected(params), "canonical greedy controls are accepted");

    params.temperature = -1.0F;
    check(sampling_params_are_rejected(params), "negative temperature is rejected");

    params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    params.top_p = 2.0F;
    check(sampling_params_are_rejected(params), "top-p above one is rejected");

    params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    params.min_p = -1.0F;
    check(sampling_params_are_rejected(params), "negative min-p is rejected");

    params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    params.repetition_penalty = 1.1F;
    check(sampling_params_are_rejected(params), "unimplemented repetition penalty is rejected");

    params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    params.top_k = 0;
    params.top_p = 0.0F;
    check(!sampling_params_are_rejected(params), "documented top-p zero greedy mode is accepted");
}

void test_eos_ids_are_range_checked() {
    trtmc::TextGenerationConfig config;
    config.eos_token_id = 8;
    auto params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    check(sampling_params_are_rejected(params), "request EOS at vocabulary size is rejected");

    params.eos_token_ids = {-1};
    check(sampling_params_are_rejected(params), "negative model EOS is rejected");
}

void test_non_finite_logits_fail_closed() {
    trtmc::TextGenerationConfig config;
    const auto params = trtmc::k2_horizon_sampling_params_from_config(config, -1);
    auto sampler = trtmc::create_k2_horizon_sampler(params);
    const std::vector<float> logits{0.0F, std::numeric_limits<float>::quiet_NaN()};
    bool rejected = false;
    try {
        (void)sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "non-finite logits are rejected");
}

bool request_is_rejected(const trtmc::TextGenerationConfig& config) {
    try {
        trtmc::k2_horizon_validate_generate_config(config);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_unsupported_generation_controls_fail_closed() {
    trtmc::TextGenerationConfig valid;
    trtmc::k2_horizon_validate_generate_config(valid);

    auto config = valid;
    config.text_generation_mode = "diffusion";
    check(request_is_rejected(config), "non-AR generation mode is rejected");

    config = valid;
    config.use_chat_template = true;
    check(request_is_rejected(config), "chat template request is rejected");

    config = valid;
    config.lora_adapter_id = "adapter";
    check(request_is_rejected(config), "LoRA request is rejected");

    config = valid;
    config.guidance_scale = 1.0F;
    check(request_is_rejected(config), "non-text generation controls are rejected");

    config = valid;
    config.max_new_tokens = -1;
    check(request_is_rejected(config), "negative generation length is rejected");

    config = valid;
    config.max_new_tokens = 0;
    check(!request_is_rejected(config), "zero generation length is accepted as a no-op");
}

void test_token_ids_are_range_checked() {
    trtmc::k2_horizon_validate_generation_inputs({0, 7}, 1, 8);

    bool negative_rejected = false;
    try {
        trtmc::k2_horizon_validate_generation_inputs({-1}, 1, 8);
    } catch (const std::invalid_argument&) {
        negative_rejected = true;
    }
    check(negative_rejected, "negative token ID is rejected");

    bool upper_bound_rejected = false;
    try {
        trtmc::k2_horizon_validate_generation_inputs({8}, 1, 8);
    } catch (const std::invalid_argument&) {
        upper_bound_rejected = true;
    }
    check(upper_bound_rejected, "token ID at vocabulary size is rejected");

    bool empty_rejected = false;
    try {
        trtmc::k2_horizon_validate_generation_inputs({}, 1, 8);
    } catch (const std::invalid_argument&) {
        empty_rejected = true;
    }
    check(empty_rejected, "empty prompt with requested generation is rejected");

    trtmc::k2_horizon_validate_generation_inputs({}, 0, 8);
}

} // namespace

int main() {
    test_greedy_ties_choose_the_lowest_index();
    test_all_model_eos_ids_are_preserved();
    test_non_greedy_requests_fail_closed();
    test_sampling_controls_are_validated_before_greedy_decode();
    test_eos_ids_are_range_checked();
    test_non_finite_logits_fail_closed();
    test_unsupported_generation_controls_fail_closed();
    test_token_ids_are_range_checked();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All K2-Horizon sampler tests passed.\n";
    return 0;
}
