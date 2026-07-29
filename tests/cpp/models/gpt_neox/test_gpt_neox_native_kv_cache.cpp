/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/gpt_neox/kv_cache.h"
#include "runtime/models/gpt_neox/pipeline.h"

namespace {

bool rejects_removed_generation_mode_before_execution() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return false;

    bool passed = false;
    {
        auto prefill_trace = std::make_shared<trtmc::test::NativeKvTrace>();
        auto decode_trace = std::make_shared<trtmc::test::NativeKvTrace>();
        auto prefill = std::make_unique<trtmc::test::NativeKvModuleStub>(
            stream, 1, 11, 1, 2, trtmc::DType::kFloat16, true, prefill_trace);
        auto decoder = std::make_unique<trtmc::test::NativeKvModuleStub>(
            stream, 1, 11, 1, 2, trtmc::DType::kFloat16, true, decode_trace);
        auto cache =
            std::make_unique<trtmc::GptNeoxKvCache>(1, 11, 2, stream, trtmc::DType::kFloat16);

        trtmc::GptNeoxTextGenConfig config;
        config.vocab_size = 16;
        config.disable_cuda_graph = true;
        config.prefill_max_length = 4;
        config.num_layers = 1;
        config.kv_dim = 2;
        std::vector<trtmc::GptNeoxTextGenerationPipeline::DecoderContext> decoders;
        decoders.push_back({11, std::move(decoder)});
        trtmc::GptNeoxTextGenerationPipeline pipeline(std::move(decoders), std::move(cache), config,
                                                      stream, nullptr, "", nullptr,
                                                      std::move(prefill));

        trtmc::GenerateConfig request;
        request.max_new_tokens = 0;
        request.text_generation_mode = "diffusion";
        bool rejected = false;
        try {
            (void)pipeline.generate_ids({1}, request);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        passed = rejected && prefill_trace->calls.empty() && decode_trace->calls.empty();
    }
    cudaStreamDestroy(stream);
    return passed;
}

} // namespace

int main() {
    int failures = trtmc::test::run_native_kv_contract_tests<
        trtmc::GptNeoxTextGenerationPipeline, trtmc::GptNeoxKvCache, trtmc::GptNeoxTextGenConfig,
        trtmc::DType::kFloat16, trtmc::test::LegacyKvPolicy::kRejected>("GPT-NeoX");
    if (!rejects_removed_generation_mode_before_execution()) {
        ++failures;
        std::cerr << "FAIL [GPT-NeoX]: removed generation mode executes or returns silently\n";
    }
    return failures;
}
