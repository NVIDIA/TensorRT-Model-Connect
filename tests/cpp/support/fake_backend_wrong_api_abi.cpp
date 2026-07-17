/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

// Simulates a backend whose C++ interface/layout belongs to another release.
extern "C" std::uint32_t trtmc_backend_api_abi_version() {
    return 0;
}
