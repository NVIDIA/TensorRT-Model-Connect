/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

// Represents a model DSO built against the previously qualified v1 C++
// plugin boundary. The v2 loader must reject it before registration.
extern "C" std::uint32_t trtmc_model_plugin_api_abi_version() {
    return 1U;
}

extern "C" const char* trtmc_model_plugin_id() {
    return "test_model_plugin_api_abi_v1";
}

extern "C" void trtmc_register_model_plugin(void*) {}
