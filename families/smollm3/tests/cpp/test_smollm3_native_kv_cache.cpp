/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/smollm3/runtime/kv_cache.h"
#include "families/smollm3/runtime/pipeline.h"
#include "families/smollm3/tests/cpp/native_kv_cache_contract_test.h"

#include <filesystem>
#include <iostream>

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    return trtmc::test::run_native_kv_contract_tests<
        trtmc::SmolLM3TextGenerationPipeline, trtmc::SmolLM3KvCache, trtmc::SmolLM3TextGenConfig>(
        "SmolLM3");
}
