/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/sampler.h"
#include "trtmc/pipeline.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

trtmc::Lfm2MoeSampleResult sample_once(const std::vector<float>& logits,
                                       const trtmc::Lfm2MoeSamplingParams& params) {
    auto sampler = trtmc::create_lfm2_moe_sampler(params);
    return sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params, {});
}

void test_hf_repetition_penalty() {
    const float logits[] = {0.5F, 4.0F, -1.0F, 3.5F};
    const auto adjusted =
        trtmc::lfm2_moe_apply_repetition_penalty(logits, 4, 2.0F, {1, 2, 1, -1, 99});
    check(adjusted.size() == 4, "penalty preserves vocabulary");
    check(std::abs(adjusted[0] - 0.5F) < 1.0e-6F, "unseen token unchanged");
    check(std::abs(adjusted[1] - 2.0F) < 1.0e-6F, "positive seen score divided once");
    check(std::abs(adjusted[2] + 2.0F) < 1.0e-6F, "negative seen score multiplied once");

    trtmc::Lfm2MoeSamplingParams params;
    params.repetition_penalty = 2.0F;
    auto sampler = trtmc::create_lfm2_moe_sampler(params);
    const auto result = sampler->sample(logits, 4, params, {1, 2});
    check(result.token_id == 3, "history penalty is applied before greedy selection");
}

void test_lfm2_moe_top_k_default_resolution() {
    trtmc::GenerateConfig request;
    request.min_p = 0.15F;
    request.temperature = 0.3F;
    const auto defaults = trtmc::lfm2_moe_sampling_params_from_config(request, {7});
    check(defaults.top_k == 1, "generic request retains its source value");
    check(trtmc::lfm2_moe_resolve_top_k(defaults) == 1,
          "generic top_k remains authoritative when other sampling knobs are active");

    request.top_k = 0;
    const auto full_vocab = trtmc::lfm2_moe_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_moe_resolve_top_k(full_vocab) == 0,
          "explicit top_k zero keeps full vocabulary");

    request.top_k = -1;
    const auto negative_full_vocab = trtmc::lfm2_moe_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_moe_resolve_top_k(negative_full_vocab) == -1,
          "negative top_k keeps the C++ API full-vocabulary contract");

    request.top_k = 17;
    const auto explicit_k = trtmc::lfm2_moe_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_moe_resolve_top_k(explicit_k) == 17,
          "explicit positive top_k is authoritative");

    request.top_k = 50;
    const auto model_card_k = trtmc::lfm2_moe_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_moe_resolve_top_k(model_card_k) == 50,
          "explicit model-card top_k 50 is preserved");
}

void test_eos_and_seeded_sampling() {
    trtmc::Lfm2MoeSamplingParams params;
    params.temperature = 0.3F;
    params.min_p = 0.15F;
    params.seed = 123;
    params.eos_token_ids = {2, 7};
    const float logits[] = {1.0F, 0.5F, 3.0F, -2.0F};
    auto first = trtmc::create_lfm2_moe_sampler(params);
    auto second = trtmc::create_lfm2_moe_sampler(params);
    const auto a = first->sample(logits, 4, params, {});
    const auto b = second->sample(logits, 4, params, {});
    check(a.token_id == b.token_id, "seeded sampling is reproducible");
    check(trtmc::lfm2_moe_is_eos_token(params, 2) && trtmc::lfm2_moe_is_eos_token(params, 7),
          "all configured EOS ids stop generation");
}

void test_distribution_filters_and_temperature() {
    const std::vector<float> logits = {2.0F, 1.0F, 0.0F};
    trtmc::Lfm2MoeSamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.seed = 3;

    const auto unfiltered = sample_once(logits, params);
    check(unfiltered.token_id == 1, "seed samples the second top-k candidate");
    check(std::abs(unfiltered.logprob + 1.3132617F) < 1.0e-6F,
          "sample log probability uses the filtered distribution");

    params.temperature = 0.5F;
    check(sample_once(logits, params).token_id == 0, "temperature scales logits before sampling");

    params.temperature = 1.0F;
    params.top_k = 1;
    const auto top_k = sample_once(logits, params);
    check(top_k.token_id == 0, "top-k limits the candidate set");
    check(std::abs(top_k.logprob) < 1.0e-6F, "single-candidate top-k is renormalized");

    params.top_k = 2;
    params.min_p = 0.4F;
    check(sample_once(logits, params).token_id == 0,
          "min-p filters relative to the maximum probability");

    params.min_p = 0.0F;
    params.top_p = 0.7F;
    check(sample_once(logits, params).token_id == 0,
          "top-p retains the smallest prefix that reaches the threshold");
}

void test_seed_fallback_and_reset() {
    const std::vector<float> logits = {2.0F, 1.0F, 0.0F};
    trtmc::Lfm2MoeSamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.seed = 3;

    auto sampler = trtmc::create_lfm2_moe_sampler(params);
    const auto first = sampler->sample(logits.data(), 3, params, {});
    (void)sampler->sample(logits.data(), 3, params, {});
    sampler->reset();
    const auto replay = sampler->sample(logits.data(), 3, params, {});
    check(first.token_id == replay.token_id && std::abs(first.logprob - replay.logprob) < 1.0e-6F,
          "reset restores the initial seeded sequence");

    params.seed = 1;
    check(sample_once(logits, params).token_id == 0, "the request seed controls sampling");

    params.seed = -1;
    const auto fallback = sample_once(logits, params);
    params.seed = 42;
    const auto explicit_fallback = sample_once(logits, params);
    check(fallback.token_id == explicit_fallback.token_id &&
              std::abs(fallback.logprob - explicit_fallback.logprob) < 1.0e-6F,
          "an unspecified seed uses the documented deterministic fallback");

    params.seed = 0;
    const auto zero_seed = sample_once(logits, params);
    params.seed = 1;
    const auto nonzero_state = sample_once(logits, params);
    check(zero_seed.token_id == nonzero_state.token_id &&
              std::abs(zero_seed.logprob - nonzero_state.logprob) < 1.0e-6F,
          "zero seed maps to a nonzero generator state");
}

} // namespace

int main() {
    test_hf_repetition_penalty();
    test_lfm2_moe_top_k_default_resolution();
    test_eos_and_seeded_sampling();
    test_distribution_filters_and_temperature();
    test_seed_fallback_and_reset();
    return failures;
}
