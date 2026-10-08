/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if __has_include("trtmc/runtime/product_build.h")
#include "trtmc/runtime/product_build.h"
#endif

#include <cstdint>

namespace trtmc {

inline constexpr std::uint32_t kPluginDescriptorVersion = 1;
inline constexpr const char* kPluginDescriptorSymbol = "trtmc_plugin_descriptor_v1";

#ifdef TRTMC_BUILD_ID
inline constexpr const char* kPluginBuildId = TRTMC_BUILD_ID;
#else
// Installed headers remain consumable, but plugins built outside a coordinated
// TRTMC product build cannot accidentally match a configured runtime.
inline constexpr const char* kPluginBuildId = "unconfigured";
#endif

enum class PluginKind : std::uint32_t {
    kBackend = 1,
    kFamily = 2,
    kRuntimeExtension = 3,
};

// Model-agnostic identity returned by every backend, family, and extension DSO.
// V1 is an exact descriptor layout, not C++ ABI compatibility negotiation.
struct PluginDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t descriptor_version;
    PluginKind kind;
    const char* id;
    const char* build_id;
};

using PluginDescriptorFn = const PluginDescriptorV1* (*)() noexcept;

} // namespace trtmc

extern "C" const trtmc::PluginDescriptorV1* trtmc_plugin_descriptor_v1() noexcept;
extern "C" const char* trtmc_core_build_id() noexcept;
extern "C" const char* trtmc_runtime_build_id() noexcept;

#define TRTMC_DEFINE_PLUGIN_DESCRIPTOR_V1(plugin_kind, plugin_id)                                  \
    extern "C" const trtmc::PluginDescriptorV1* trtmc_plugin_descriptor_v1() noexcept {            \
        static const trtmc::PluginDescriptorV1 descriptor{                                         \
            sizeof(trtmc::PluginDescriptorV1), trtmc::kPluginDescriptorVersion, plugin_kind,       \
            plugin_id, trtmc::kPluginBuildId};                                                     \
        return &descriptor;                                                                        \
    }
