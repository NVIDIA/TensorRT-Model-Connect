/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/smollm3/kv_cache.h"
#include "runtime/models/smollm3/pipeline.h"
#include "runtime/models/smollm3/plugin_helpers.h"

namespace {

int test_dynamic_prefill_uses_runtime_cache_rows() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    int failures = 0;
    {
        trtmc::Smollm3KvCache cache(1, 1000, 2, stream, trtmc::DType::kFloat16);
        trtmc::test::NativeKvModuleStub prefill(stream, 1, 131072, 1, 2, trtmc::DType::kFloat16,
                                                /*native=*/false, nullptr, 4, 16,
                                                /*dynamic_legacy_cache=*/true);

        cache.bind_cache_inputs(prefill);
        if (!trtmc::cache_input_supports_runtime_rows(prefill, "cache_k_0")) {
            std::cerr << "FAIL [SmolLM3]: dynamic metadata survives positive active tensor shape\n";
            ++failures;
        }
        if (prefill.bound_shape("cache_k_0") != std::vector<int64_t>{1000, 2} ||
            prefill.bound_shape("cache_v_0") != std::vector<int64_t>{1000, 2}) {
            std::cerr << "FAIL [SmolLM3]: dynamic prefill binds runtime-sized cache rows\n";
            ++failures;
        }

        trtmc::TensorMap inputs;
        cache.prepare_step(inputs, 4);
        if (inputs.at("attention_mask").shape != std::vector<int64_t>{4, 1004}) {
            std::cerr << "FAIL [SmolLM3]: dynamic prefill mask uses runtime cache rows\n";
            ++failures;
        }
    }
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace

int main() {
    return trtmc::test::run_native_kv_contract_tests<trtmc::Smollm3TextGenerationPipeline,
                                                     trtmc::Smollm3KvCache,
                                                     trtmc::Smollm3TextGenConfig>("SmolLM3") +
           test_dynamic_prefill_uses_runtime_cache_rows();
}
