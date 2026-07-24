/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace trtmc::runtime_kv {

// Common DSO handshake for the selected runtime-KV implementation. This is
// intentionally independent of any individual creator ABI: NativeKvAppend
// remains an ABI-v1 negative fixture while NativeContiguousAttention is v2.
inline constexpr int32_t kRuntimeKvPluginDsoAbi = 2;
inline constexpr std::uint64_t kRuntimeKvCapabilityCudnnSdpa = 1ULL << 0;

} // namespace trtmc::runtime_kv

extern "C" int32_t trtmc_runtime_kv_plugin_abi_version() noexcept;
extern "C" std::uint64_t trtmc_runtime_kv_plugin_capabilities() noexcept;
// Versioned, independently detected runtime facts for build/runtime admission.
// The returned UTF-8 JSON pointer remains valid until the next call on the
// same thread. It never contains values supplied by bundle metadata.
extern "C" const char* trtmc_runtime_kv_plugin_runtime_stack_json_v1() noexcept;
extern "C" void trtmc_runtime_kv_plugin_force_link() noexcept;
