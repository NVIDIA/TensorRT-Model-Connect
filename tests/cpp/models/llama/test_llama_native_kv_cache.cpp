/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../native_kv_cache_contract_test.h"
#include "runtime/models/llama/kv_cache.h"

int main() {
    return trtmc::test::run_native_kv_cache_contract_test<trtmc::LlamaKvCache>(131072, "Llama");
}
