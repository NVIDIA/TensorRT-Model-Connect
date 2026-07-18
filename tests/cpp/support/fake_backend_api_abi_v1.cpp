/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

// Simulates the previously qualified backend ABI v1 without including the
// current IBackend definition. The v2 loader must reject it before resolving
// or calling either v1 factory symbol.
extern "C" std::uint32_t trtmc_backend_api_abi_version() {
    return 1U;
}

extern "C" void* trtmc_create_backend_v1() {
    return nullptr;
}

extern "C" void trtmc_destroy_backend_v1(void*) {}
