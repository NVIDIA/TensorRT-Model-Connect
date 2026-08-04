/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/deepseek_ocr/kv_cache.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [DeepSeek-OCR]: " << message << '\n';
            ++failures;
        }
    };

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    {
        trtmc::DeepseekOcrKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub module(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16);
        module.set_tensor("key_value_lengths", {2}, trtmc::DType::kInt32);
        bool rejected = false;
        try {
            cache.bind_to(module);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "rejects invalid scalar shape");
    }

    {
        trtmc::DeepseekOcrKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub module(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16);
        module.set_tensor("cache_k_0", {1, 1, 10, 2}, trtmc::DType::kBFloat16);
        bool rejected = false;
        try {
            cache.bind_to(module);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "rejects mismatched full capacity");
    }

    {
        trtmc::DeepseekOcrKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub prefill(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub decode(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16);
        cache.bind_to(prefill);
        cache.bind_to(decode);
        check(cache.device_memory_bytes() == 88, "allocates one full-capacity K/V pair");
        check(prefill.device_ptr("cache_k_0") != nullptr &&
                  prefill.device_ptr("present_k_0") == prefill.device_ptr("cache_k_0") &&
                  decode.device_ptr("cache_k_0") == prefill.device_ptr("cache_k_0"),
              "shares caller-owned cache across split engines");

        trtmc::TensorMap inputs;
        cache.prepare_step(inputs, 4);
        check(inputs.count("attention_mask") == 0 &&
                  trtmc::test::scalar(inputs, "cache_write_indices") == 0 &&
                  trtmc::test::scalar(inputs, "key_value_lengths") == 4,
              "supplies write index and active length without a dense mask");
        cache.advance(4);
        inputs.clear();
        cache.prepare_step(inputs, 7);
        check(trtmc::test::scalar(inputs, "cache_write_indices") == 4 &&
                  trtmc::test::scalar(inputs, "key_value_lengths") == 11,
              "chunked prefill advances absolute cache positions");
        cache.advance(7);
        bool overflow = false;
        try {
            cache.prepare_step(inputs);
        } catch (const std::runtime_error&) {
            overflow = true;
        }
        check(overflow && cache.position() == 11,
              "rejects an over-capacity write without progression");
    }

    {
        trtmc::DeepseekOcrKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub legacy(stream, 1, 11, 1, 2, trtmc::DType::kBFloat16, false);
        bool rejected = false;
        try {
            cache.bind_to(legacy);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "fails closed on a legacy KV engine");
    }

    {
        trtmc::DeepseekOcrKvCache rank_cache(1, 16, 128, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub rank_engine(stream, 1, 16, 1, 128, trtmc::DType::kBFloat16);
        rank_cache.bind_to(rank_engine);
        check(rank_cache.ok() && rank_cache.device_memory_bytes() == 8192,
              "TP rank allocates only local KV heads");
    }

    cudaStreamDestroy(stream);
    return failures;
}
