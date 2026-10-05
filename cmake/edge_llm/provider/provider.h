/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>

namespace trtmc::edge_llm {
/// No vendor classes, STL ownership or exceptions cross this versioned DSO boundary.
/// All returned strings are provider-owned, valid until its next call on that thread/handle.
struct ProviderV1 {
    std::uint32_t abi;
    std::uint32_t size;
    const char* version;
    void* (*open)(const char* worker, const char* descriptor);
    const char* (*call)(void* handle, const char* json);
    void (*close)(void* handle);
    const char* (*error)();
};
using GetProviderV1 = const ProviderV1* (*)();
} // namespace trtmc::edge_llm
