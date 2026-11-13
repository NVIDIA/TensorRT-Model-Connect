/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gpt_oss/runtime/eos_tokens.h"
#include "families/gpt_oss/runtime/sampler.h"
#include "trtmc/task.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_config() {
    const auto scalar = trtmc::gpt_oss::require_eos_tokens({{"eos_token_id", 0}}, 201088);
    check(scalar == std::vector<std::int32_t>{0}, "scalar zero EOS");
    const auto ids =
        trtmc::gpt_oss::require_eos_tokens({{"eos_token_id", {200002, 199999, 200012}}}, 201088);
    check(ids == std::vector<std::int32_t>({200002, 199999, 200012}), "all checkpoint EOS IDs");
    const std::vector<nlohmann::json> invalid{
        nlohmann::json::object(),
        {{"eos_token_id", nlohmann::json::array()}},
        {{"eos_token_id", nullptr}},
        {{"eos_token_id", true}},
        {{"eos_token_id", -1}},
        {{"eos_token_id", 201088}},
        {{"eos_token_id", {200002, -1}}},
        {{"eos_token_id", {200002, 201088}}},
        {{"eos_token_id", {200002, 1.5}}},
        {{"eos_token_id", {200002, false}}},
        {{"eos_token_id", {200002, "200012"}}},
        {{"eos_token_id", std::numeric_limits<std::uint64_t>::max()}},
    };
    for (const auto& config : invalid) {
        bool rejected = false;
        try {
            (void)trtmc::gpt_oss::require_eos_tokens(config, 201088);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid EOS configuration accepted");
    }
}

void test_samplers() {
    const std::vector<std::int32_t> defaults{200002, 199999, 200012};
    for (int mode = 0; mode < 3; ++mode) {
        trtmc::TextGenerationConfig request;
        if (mode != 0) {
            request.seed = 7;
            request.top_k = 1;
            request.temperature = mode == 1 ? 1.0F : 0.0F;
        }
        auto params =
            trtmc::gpt_oss_sampling_params_from_config(request, defaults.front(), defaults);
        auto sampler = trtmc::create_gpt_oss_sampler(params);
        for (const auto token : {200002, 199999, 200012, 17}) {
            std::vector<float> logits(201088, -10.0F);
            logits[static_cast<std::size_t>(token)] = 10.0F;
            const auto result = sampler->sample(logits.data(), logits.size(), params);
            check(result.token_id == token, "EOS handling changed token selection");
            check(result.is_eos == (token != 17), "checkpoint EOS was ignored");
        }
        request.eos_token_id = 0;
        params = trtmc::gpt_oss_sampling_params_from_config(request, defaults.front(), defaults);
        check(params.eos_token_ids == std::vector<std::int32_t>{0},
              "override did not replace EOS set");
        for (const auto token : {0, 200002, 200012}) {
            std::vector<float> logits(201088, -10.0F);
            logits[static_cast<std::size_t>(token)] = 10.0F;
            const auto result = sampler->sample(logits.data(), logits.size(), params);
            check(result.is_eos == (token == 0), "scalar zero override ignored");
        }
        check(sampler->sample(nullptr, 0, params).is_eos, "empty-logit zero EOS behavior");
        request.eos_token_id = -1;
        params = trtmc::gpt_oss_sampling_params_from_config(request, 0);
        check(sampler->sample(nullptr, 0, params).is_eos, "legacy scalar zero EOS behavior");
    }
}

} // namespace

int main() {
    try {
        test_config();
        test_samplers();
        std::cout << "GPT-OSS EOS parser and sampler contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
