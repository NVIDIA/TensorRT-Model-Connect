/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/runtime/plugin_abi.h"
#include "trtmc/task.h"

#include <cstdint>

namespace trtmc {

class IBackend;

struct FamilyContext {
    const BundleReader& reader;
    IBackend& backend;
    std::uint64_t kv_cache_size_bytes{0};
};

using CreateFamilyFn = ITask* (*)(const FamilyContext& context);

inline constexpr const char* kCreateFamilySymbol = "trtmc_create_family";

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context);

#define TRTMC_DEFINE_FAMILY_PLUGIN_V1(family_id)                                                   \
    TRTMC_DEFINE_PLUGIN_DESCRIPTOR_V1(::trtmc::PluginKind::kFamily, family_id)
