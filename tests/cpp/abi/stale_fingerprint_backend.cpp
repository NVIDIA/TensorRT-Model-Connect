/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_memory_backend.h"

#include <cstdint>

namespace {

std::uint32_t g_create_calls = 0;

} // namespace

extern "C" std::int32_t
trtmc_backend_query_abi_contract_v2(trtmc::BackendDsoAbiContractV2* contract,
                                    std::size_t contract_size) noexcept {
    if (contract == nullptr || contract_size < sizeof(*contract))
        return -1;
    *contract = trtmc::make_runtime_memory_backend_dso_abi_contract_v2(0);
    contract->interface_fingerprint ^= 1;
    return 0;
}

extern "C" trtmc::IBackend* trtmc_create_backend() {
    ++g_create_calls;
    return nullptr;
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend*) {}

extern "C" std::uint32_t trtmc_test_stale_backend_create_calls() {
    return g_create_calls;
}
