/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2/sampler.h"
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

void test_hf_repetition_penalty() {
    const float logits[] = {0.5F, 4.0F, -1.0F, 3.5F};
    const auto adjusted = trtmc::lfm2_apply_repetition_penalty(logits, 4, 2.0F, {1, 2, 1, -1, 99});
    check(adjusted.size() == 4, "penalty preserves vocabulary");
    check(std::abs(adjusted[0] - 0.5F) < 1.0e-6F, "unseen token unchanged");
    check(std::abs(adjusted[1] - 2.0F) < 1.0e-6F, "positive seen score divided once");
    check(std::abs(adjusted[2] + 2.0F) < 1.0e-6F, "negative seen score multiplied once");

    trtmc::Lfm2SamplingParams params;
    params.repetition_penalty = 2.0F;
    auto sampler = trtmc::create_lfm2_sampler(params);
    const auto result = sampler->sample(logits, 4, params, {1, 2});
    check(result.token_id == 3, "history penalty is applied before greedy selection");
}

void test_lfm2_top_k_default_resolution() {
    trtmc::GenerateConfig request;
    request.min_p = 0.15F;
    request.temperature = 0.3F;
    const auto defaults = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(defaults.top_k == 1, "generic request retains its source value");
    check(trtmc::lfm2_resolve_top_k(defaults) == 1,
          "generic top_k remains authoritative when other sampling knobs are active");

    request.top_k = 0;
    const auto full_vocab = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(full_vocab) == 0, "explicit top_k zero keeps full vocabulary");

    request.top_k = -1;
    const auto negative_full_vocab = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(negative_full_vocab) == -1,
          "negative top_k keeps the C++ API full-vocabulary contract");

    request.top_k = 17;
    const auto explicit_k = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(explicit_k) == 17, "explicit positive top_k is authoritative");

    request.top_k = 50;
    const auto model_card_k = trtmc::lfm2_sampling_params_from_config(request, {7});
    check(trtmc::lfm2_resolve_top_k(model_card_k) == 50,
          "explicit model-card top_k 50 is preserved");
}

void test_eos_and_seeded_sampling() {
    trtmc::Lfm2SamplingParams params;
    params.temperature = 0.3F;
    params.min_p = 0.15F;
    params.seed = 123;
    params.eos_token_ids = {2, 7};
    const float logits[] = {1.0F, 0.5F, 3.0F, -2.0F};
    auto first = trtmc::create_lfm2_sampler(params);
    auto second = trtmc::create_lfm2_sampler(params);
    const auto a = first->sample(logits, 4, params, {});
    const auto b = second->sample(logits, 4, params, {});
    check(a.token_id == b.token_id, "seeded sampling is reproducible");
    check(trtmc::lfm2_is_eos_token(params, 2) && trtmc::lfm2_is_eos_token(params, 7),
          "all configured EOS ids stop generation");
}

} // namespace

int main() {
    test_hf_repetition_penalty();
    test_lfm2_top_k_default_resolution();
    test_eos_and_seeded_sampling();
    return failures;
}
