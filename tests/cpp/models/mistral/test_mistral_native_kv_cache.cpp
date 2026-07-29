/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/mistral/kv_cache.h"
#include "runtime/models/mistral/pipeline.h"

namespace {

int test_generation_modes() {
    using Cache = trtmc::MistralKvCache;
    using Config = trtmc::MistralTextGenConfig;
    using Pipeline = trtmc::MistralTextGenerationPipeline;
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
        config.num_layers = 1;
        config.kv_dim = 2;

        auto prefill = std::make_unique<Module>(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16, true);
        auto decoder = std::make_unique<Module>(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16, true);
        auto cache = std::make_unique<Cache>(1, 11, 2, stream, trtmc::DType::kBFloat16);
        std::vector<Pipeline::DecoderContext> decoders;
        decoders.push_back(Pipeline::DecoderContext{11, std::move(decoder)});
        Pipeline pipeline(std::move(decoders), std::move(cache), std::move(config), stream, nullptr,
                          "", nullptr, std::move(prefill));

        trtmc::GenerateConfig request;
        request.max_new_tokens = 0;
        for (const std::string& mode : {"", "auto", "ar", "autoregressive"}) {
            request.text_generation_mode = mode;
            try {
                pipeline.generate_ids({1}, request);
            } catch (const std::runtime_error&) {
                std::cerr << "FAIL [Mistral]: rejected supported generation mode '" << mode
                          << "'\n";
                ++failures;
            }
        }
        for (const std::string& mode :
             {"diffusion", "dlm", "linear_spec", "linear-speculation", "unknown"}) {
            request.text_generation_mode = mode;
            bool rejected = false;
            try {
                pipeline.generate_ids({1}, request);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) {
                std::cerr << "FAIL [Mistral]: accepted unsupported generation mode '" << mode
                          << "'\n";
                ++failures;
            }
        }
        request.text_generation_mode = "diffusion";
        bool empty_rejected = false;
        try {
            pipeline.generate_ids({}, request);
        } catch (const std::runtime_error&) {
            empty_rejected = true;
        }
        if (!empty_rejected) {
            std::cerr << "FAIL [Mistral]: empty no-op bypassed generation mode validation\n";
            ++failures;
        }
    }
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace

int main() {
    int failures = trtmc::test::run_native_kv_contract_tests<
        trtmc::MistralTextGenerationPipeline, trtmc::MistralKvCache, trtmc::MistralTextGenConfig,
        trtmc::DType::kBFloat16, trtmc::test::LegacyKvPolicy::kRejected>("Mistral");
    failures += test_generation_modes();
    return failures;
}
