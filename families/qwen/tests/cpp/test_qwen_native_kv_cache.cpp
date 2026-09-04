/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/kv_cache.h"
#include "families/qwen/runtime/pipeline.h"
#include "families/qwen/tests/cpp/native_kv_cache_contract_test.h"

#include <filesystem>
#include <iostream>

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    return trtmc::test::run_native_kv_contract_tests<trtmc::QwenTextGenerationPipeline,
                                                     trtmc::QwenKvCache, trtmc::QwenTextGenConfig>(
        "Qwen");
}
