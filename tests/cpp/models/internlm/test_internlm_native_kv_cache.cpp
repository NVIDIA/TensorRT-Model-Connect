/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/internlm/kv_cache.h"
#include "runtime/models/internlm/pipeline.h"
#include "runtime/models/internlm/plugin_helpers.h"

#include <array>
#include <iostream>

namespace {

using ShapeTriplet = std::array<int32_t, 3>;

bool rejects_profile(cudaStream_t stream, trtmc::InternlmEngineRole role,
                     ShapeTriplet token_profile, ShapeTriplet position_profile, int32_t capacity,
                     int32_t profile_idx = 0, int32_t profile_count = 1) {
    auto trace = std::make_shared<trtmc::test::NativeKvTrace>();
    trtmc::test::NativeKvModuleStub module(stream, 1, capacity, 1, 2, trtmc::DType::kBFloat16, true,
                                           std::move(trace));
    module.set_profile_shape("token_id", {token_profile[0]}, {token_profile[1]},
                             {token_profile[2]});
    module.set_profile_shape("position_id", {position_profile[0]}, {position_profile[1]},
                             {position_profile[2]});
    module.set_profile_metadata(profile_idx, profile_count);
    try {
        (void)trtmc::validate_internlm_native_sequence_profile(module, "token_id", "position_id",
                                                               capacity, role);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

int run_profile_contract_tests() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [Internlm profile]: " << message << '\n';
            ++failures;
        }
    };

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    check(!rejects_profile(stream, trtmc::InternlmEngineRole::kDecode, {1, 1, 1}, {1, 1, 1}, 11),
          "accepts decode min=opt=max=1");
    check(!rejects_profile(stream, trtmc::InternlmEngineRole::kPrefill, {1, 4, 8}, {1, 4, 8}, 11),
          "accepts bounded prefill min=1<opt<=max");
    check(rejects_profile(stream, trtmc::InternlmEngineRole::kDecode, {1, 2, 2}, {1, 2, 2}, 11),
          "rejects a decode-shaped profile with opt/max above one");
    check(rejects_profile(stream, trtmc::InternlmEngineRole::kPrefill, {1, 1, 1}, {1, 1, 1}, 11),
          "rejects a decode profile in the prefill slot");
    check(rejects_profile(stream, trtmc::InternlmEngineRole::kPrefill, {1, 4, 8}, {1, 4, 7}, 11),
          "rejects mismatched token and position profiles");
    check(rejects_profile(stream, trtmc::InternlmEngineRole::kPrefill, {1, 4, 12}, {1, 4, 12}, 11),
          "rejects a prefill profile above KV capacity");
    check(rejects_profile(stream, trtmc::InternlmEngineRole::kDecode, {1, 1, 1}, {1, 1, 1}, 11, 1),
          "rejects a module bound to a nonzero profile");
    check(
        rejects_profile(stream, trtmc::InternlmEngineRole::kDecode, {1, 1, 1}, {1, 1, 1}, 11, 0, 2),
        "rejects a multi-profile engine");

    cudaStreamDestroy(stream);
    return failures;
}

int run_generation_mode_contract_tests() {
    using Cache = trtmc::InternlmKvCache;
    using Config = trtmc::InternlmTextGenConfig;
    using Pipeline = trtmc::InternlmTextGenerationPipeline;
    using Module = trtmc::test::NativeKvModuleStub;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    int failures = 0;
    {
        Config config;
        config.vocab_size = 16;
        config.disable_cuda_graph = true;
        config.prefill_max_length = 4;
        auto prefill = std::make_unique<Module>(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16, true);
        auto decoder = std::make_unique<Module>(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16, true);
        auto cache = std::make_unique<Cache>(1, 11, 2, stream, trtmc::DType::kBFloat16);
        std::vector<Pipeline::DecoderContext> decoders;
        decoders.push_back(Pipeline::DecoderContext{std::move(decoder)});
        Pipeline pipeline(std::move(decoders), std::move(cache), std::move(config), stream, nullptr,
                          "", nullptr, std::move(prefill));

        trtmc::GenerateConfig request;
        request.max_new_tokens = 0;
        for (const std::string& mode : {"", "auto", "ar", "autoregressive"}) {
            request.text_generation_mode = mode;
            try {
                (void)pipeline.generate_ids({1}, request);
            } catch (const std::runtime_error&) {
                std::cerr << "FAIL [InternLM mode]: rejected supported mode '" << mode << "'\n";
                ++failures;
            }
        }
        for (const std::string& mode :
             {"diffusion", "dlm", "linear_spec", "linear-speculation", "unknown"}) {
            request.text_generation_mode = mode;
            bool rejected = false;
            try {
                (void)pipeline.generate_ids({1}, request);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) {
                std::cerr << "FAIL [InternLM mode]: accepted unsupported mode '" << mode << "'\n";
                ++failures;
            }
        }

        request.text_generation_mode = "diffusion";
        bool empty_rejected = false;
        try {
            (void)pipeline.generate_ids({}, request);
        } catch (const std::runtime_error&) {
            empty_rejected = true;
        }
        if (!empty_rejected) {
            std::cerr << "FAIL [InternLM mode]: empty no-op bypassed mode validation\n";
            ++failures;
        }
    }

    cudaStreamDestroy(stream);
    return failures;
}

} // namespace

int main() {
    return trtmc::test::run_native_kv_contract_tests<
               trtmc::InternlmTextGenerationPipeline, trtmc::InternlmKvCache,
               trtmc::InternlmTextGenConfig, trtmc::DType::kBFloat16, false, false>("Internlm") +
           run_profile_contract_tests() + run_generation_mode_contract_tests();
}
