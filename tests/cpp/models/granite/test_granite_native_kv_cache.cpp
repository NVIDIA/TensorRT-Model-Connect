/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/granite/kv_cache.h"
#include "runtime/models/granite/pipeline.h"

int main() {
    return trtmc::test::run_native_kv_contract_tests<
        trtmc::GraniteTextGenerationPipeline, trtmc::GraniteKvCache, trtmc::GraniteTextGenConfig,
        trtmc::DType::kFloat16, trtmc::test::NativeKvLegacyPolicy::kRejects>("Granite");
}
