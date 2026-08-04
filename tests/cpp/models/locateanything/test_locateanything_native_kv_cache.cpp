/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/locateanything/kv_cache.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [LocateAnything]: " << message << '\n';
            ++failures;
        }
    };

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    check(trtmc::test::rejects_native_contract<trtmc::LocateanythingKvCache>(
              stream,
              [](auto& module) {
                  module.set_tensor("key_value_lengths", {2}, trtmc::DType::kInt32);
              }),
          "rejects invalid scalar shape");
    check(trtmc::test::rejects_native_contract<trtmc::LocateanythingKvCache>(
              stream,
              [](auto& module) {
                  module.set_tensor("cache_k_0", {1, 1, 10, 2}, trtmc::DType::kFloat16);
              }),
          "rejects mismatched full capacity");

    {
        trtmc::LocateanythingKvCache cache(1, 11, 2, stream, trtmc::DType::kFloat16);
        trtmc::test::NativeKvModuleStub prefill(stream, 1, 11, 1, 2, trtmc::DType::kFloat16);
        trtmc::test::NativeKvModuleStub decode(stream, 1, 11, 1, 2, trtmc::DType::kFloat16);
        cache.bind_cache_inputs(prefill);
        cache.bind_to(decode);
        check(cache.device_memory_bytes() == 88, "allocates only one full-capacity K/V pair");
        check(prefill.device_ptr("present_k_0") == cache.cache_k(0).data() &&
                  decode.device_ptr("present_v_0") == cache.cache_v(0).data(),
              "aliases prefill and decode present tensors to caller-owned storage");

        trtmc::TensorMap inputs;
        cache.prepare_step(inputs, 4);
        check(inputs.count("attention_mask") == 0 &&
                  trtmc::test::scalar(inputs, "cache_write_indices") == 0 &&
                  trtmc::test::scalar(inputs, "key_value_lengths") == 4,
              "uses native write-index and active-length inputs");

        std::vector<const void*> present_k{prefill.device_ptr("present_k_0")};
        std::vector<const void*> present_v{prefill.device_ptr("present_v_0")};
        cache.append_prefill_kv(present_k, present_v, 4);
        check(cache.position() == 4, "chunked prefill advances without a device copy");
    }

    {
        // A TP rank owns only its local KV head. The cache contract is based on
        // the rank-local Hkv*D width, not the global model width.
        trtmc::LocateanythingKvCache rank_cache(1, 16, 128, stream, trtmc::DType::kBFloat16);
        trtmc::test::NativeKvModuleStub rank_engine(stream, 1, 16, 1, 128, trtmc::DType::kBFloat16);
        rank_cache.bind_to(rank_engine);
        check(rank_cache.ok() && rank_cache.device_memory_bytes() == 8192,
              "TP rank allocates compact local KV storage");
    }

    cudaStreamDestroy(stream);
    return failures;
}
