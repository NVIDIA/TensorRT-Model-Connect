/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kv_cache.h"
#include "native_kv_cache_contract_test.h"
#include "pipeline.h"

int main() {
    return trtmc::test::run_native_kv_contract_tests<
        trtmc::LlamaTextGenerationPipeline, trtmc::LlamaKvCache, trtmc::LlamaTextGenConfig>(
        "Llama");
}
