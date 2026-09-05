/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/kv_cache.h"
#include "families/llama/runtime/pipeline.h"
#include "families/llama/tests/cpp/native_kv_cache_contract_test.h"

#include <filesystem>
#include <iostream>

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    return trtmc::test::run_native_kv_contract_tests<
        trtmc::LlamaTextGenerationPipeline, trtmc::LlamaKvCache, trtmc::LlamaTextGenConfig>(
        "Llama");
}
