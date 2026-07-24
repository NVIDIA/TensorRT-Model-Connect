/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently compiled pre-handshake backend fixture.  It intentionally
// includes no current TRTMC header and exports only the legacy C factory
// symbols plus same-TensorRT metadata.  The current loader must reject it
// before trtmc_create_backend() can increment the counter.

#include <cstdint>

namespace {

std::uint32_t g_create_calls = 0;

} // namespace

extern "C" void* trtmc_create_backend() {
    ++g_create_calls;
    return nullptr;
}

extern "C" void trtmc_destroy_backend(void*) {}

extern "C" const char* trtmc_backend_abi() {
    return "11_2";
}

extern "C" const char* trtmc_backend_runtime_version() {
    return "11.2.0.113";
}

extern "C" std::uint32_t trtmc_test_stale_backend_create_calls() {
    return g_create_calls;
}
