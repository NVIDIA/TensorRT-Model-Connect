/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/k2_horizon/sampler.h"

#include "trtmc/pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc {
namespace {

bool controls_are_finite(const K2HorizonSamplingParams& params) {
    return std::isfinite(params.temperature) && std::isfinite(params.top_p) &&
           std::isfinite(params.min_p) && std::isfinite(params.repetition_penalty);
}

bool probabilities_are_in_range(const K2HorizonSamplingParams& params) {
    return params.temperature >= 0.0F && params.top_p >= 0.0F && params.top_p <= 1.0F &&
           params.min_p >= 0.0F && params.min_p <= 1.0F;
}

bool selects_greedy_decode(const K2HorizonSamplingParams& params) {
    return params.temperature == 0.0F || params.top_k == 1 || params.top_p == 0.0F;
}

void validate_greedy_controls(const K2HorizonSamplingParams& params) {
    if (!controls_are_finite(params))
        throw std::invalid_argument("K2-Horizon sampling controls must be finite");
    if (!probabilities_are_in_range(params))
        throw std::invalid_argument("K2-Horizon sampling controls are outside their legal range");
    if (params.repetition_penalty != 1.0F)
        throw std::invalid_argument("K2-Horizon does not support repetition penalty");
    if (!selects_greedy_decode(params)) {
        throw std::invalid_argument(
            "K2-Horizon currently supports deterministic greedy generation only");
    }
}

class GreedySampler final : public K2HorizonISampler {
  public:
    K2HorizonSampleResult sample(const float* logits, int32_t vocab_size,
                                 const K2HorizonSamplingParams& params) override {
        if (logits == nullptr || vocab_size <= 0)
            throw std::invalid_argument("K2-Horizon greedy sampling requires nonempty logits");
        if (!std::isfinite(logits[0]))
            throw std::runtime_error("K2-Horizon greedy sampling received non-finite logits");
        int32_t best = 0;
        for (int32_t index = 1; index < vocab_size; ++index) {
            if (!std::isfinite(logits[index]))
                throw std::runtime_error("K2-Horizon greedy sampling received non-finite logits");
            if (logits[index] > logits[best])
                best = index;
        }
        return K2HorizonSampleResult{best, k2_horizon_is_eos_token(params, best)};
    }

    const char* sampler_type() const override { return "greedy"; }
};

} // namespace

bool k2_horizon_is_eos_token(const K2HorizonSamplingParams& params, int32_t token_id) {
    if (!params.eos_token_ids.empty()) {
        return std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token_id) !=
               params.eos_token_ids.end();
    }
    return params.eos_token_id >= 0 && token_id == params.eos_token_id;
}

void k2_horizon_validate_sampling_params(const K2HorizonSamplingParams& params,
                                         int32_t vocab_size) {
    validate_greedy_controls(params);
    if (vocab_size <= 0)
        throw std::invalid_argument("K2-Horizon vocabulary size must be positive");
    for (int32_t token_id : params.eos_token_ids) {
        if (token_id < 0 || token_id >= vocab_size)
            throw std::invalid_argument("K2-Horizon EOS token is outside the model vocabulary");
    }
    if (params.eos_token_ids.empty() &&
        (params.eos_token_id < -1 || params.eos_token_id >= vocab_size)) {
        throw std::invalid_argument("K2-Horizon EOS token is outside the model vocabulary");
    }
}

K2HorizonSamplingParams
k2_horizon_sampling_params_from_config(const GenerateConfig& cfg,
                                       const std::vector<int32_t>& default_eos_token_ids) {
    K2HorizonSamplingParams params;
    params.temperature = cfg.temperature;
    params.top_k = cfg.top_k;
    params.top_p = cfg.top_p;
    params.min_p = cfg.min_p;
    params.repetition_penalty = cfg.repetition_penalty;
    params.seed = cfg.seed;
    params.eos_token_ids =
        cfg.eos_token_id >= 0 ? std::vector<int32_t>{cfg.eos_token_id} : default_eos_token_ids;
    params.eos_token_id = params.eos_token_ids.empty() ? -1 : params.eos_token_ids.front();
    return params;
}

K2HorizonSamplingParams k2_horizon_sampling_params_from_config(const GenerateConfig& cfg,
                                                               int32_t default_eos) {
    const std::vector<int32_t> defaults =
        default_eos >= 0 ? std::vector<int32_t>{default_eos} : std::vector<int32_t>{};
    return k2_horizon_sampling_params_from_config(cfg, defaults);
}

std::unique_ptr<K2HorizonISampler>
create_k2_horizon_sampler(const K2HorizonSamplingParams& params) {
    validate_greedy_controls(params);
    return std::make_unique<GreedySampler>();
}

} // namespace trtmc
