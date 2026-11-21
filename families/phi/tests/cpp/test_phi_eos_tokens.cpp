/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phi/runtime/eos_tokens.h"
#include "families/phi/runtime/sampler.h"
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
    const auto scalar = trtmc::phi::require_eos_tokens({{"eos_token_id", 0}}, 32064);
    check(scalar == std::vector<std::int32_t>{0}, "scalar zero EOS");
    const auto ids =
        trtmc::phi::require_eos_tokens({{"eos_token_id", {32000, 32001, 32007}}}, 32064);
    check(ids == std::vector<std::int32_t>({32000, 32001, 32007}), "all checkpoint EOS IDs");
    const std::vector<nlohmann::json> invalid{
        nlohmann::json::object(),
        {{"eos_token_id", nlohmann::json::array()}},
        {{"eos_token_id", nullptr}},
        {{"eos_token_id", true}},
        {{"eos_token_id", -1}},
        {{"eos_token_id", 32064}},
        {{"eos_token_id", {32000, -1}}},
        {{"eos_token_id", {32000, 32064}}},
        {{"eos_token_id", {32000, 1.5}}},
        {{"eos_token_id", {32000, false}}},
        {{"eos_token_id", {32000, "32007"}}},
        {{"eos_token_id", std::numeric_limits<std::uint64_t>::max()}},
    };
    for (const auto& config : invalid) {
        bool rejected = false;
        try {
            (void)trtmc::phi::require_eos_tokens(config, 32064);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid EOS configuration accepted");
    }
}

void test_samplers() {
    const std::vector<std::int32_t> defaults{32000, 32001, 32007};
    for (int mode = 0; mode < 3; ++mode) {
        trtmc::TextGenerationConfig request;
        if (mode != 0) {
            request.seed = 7;
            request.top_k = 1;
            request.temperature = mode == 1 ? 1.0F : 0.0F;
        }
        auto params = trtmc::phi_sampling_params_from_config(request, defaults.front(), defaults);
        auto sampler = trtmc::create_phi_sampler(params);
        for (const auto token : {32000, 32001, 32007, 17}) {
            std::vector<float> logits(32064, -10.0F);
            logits[static_cast<std::size_t>(token)] = 10.0F;
            const auto result = sampler->sample(logits.data(), logits.size(), params);
            check(result.token_id == token, "EOS handling changed token selection");
            check(result.is_eos == (token != 17), "checkpoint EOS was ignored");
        }
        request.eos_token_id = 0;
        params = trtmc::phi_sampling_params_from_config(request, defaults.front(), defaults);
        check(params.eos_token_ids == std::vector<std::int32_t>{0},
              "override did not replace EOS set");
        for (const auto token : {0, 32000, 32007}) {
            std::vector<float> logits(32064, -10.0F);
            logits[static_cast<std::size_t>(token)] = 10.0F;
            const auto result = sampler->sample(logits.data(), logits.size(), params);
            check(result.is_eos == (token == 0), "scalar zero override ignored");
        }
        check(sampler->sample(nullptr, 0, params).is_eos, "empty-logit zero EOS behavior");
        request.eos_token_id = -1;
        params = trtmc::phi_sampling_params_from_config(request, 0);
        check(sampler->sample(nullptr, 0, params).is_eos, "legacy scalar zero EOS behavior");
    }
}

} // namespace

int main() {
    try {
        test_config();
        test_samplers();
        std::cout << "Phi EOS parser and sampler contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
