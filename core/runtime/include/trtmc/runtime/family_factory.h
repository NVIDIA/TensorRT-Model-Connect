/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/task.h"

#include <cstdint>

namespace trtmc {

class IBackend;

struct FamilyContext {
    const BundleReader& reader;
    IBackend& backend;
    std::uint64_t kv_cache_size_bytes{0};
};

// A family that owns a complete external runtime can use this context instead
// of pretending that runtime implements the Engine API. Returning nullptr
// means the bundle uses the normal Engine backend factory.
struct FamilyOnlyContext {
    const BundleReader& reader;
    std::uint64_t kv_cache_size_bytes{0};
};

using CreateFamilyFn = ITask* (*)(const FamilyContext& context);
using CreateFamilyOnlyFn = ITask* (*)(const FamilyOnlyContext& context);

inline constexpr const char* kCreateFamilySymbol = "trtmc_create_family";
inline constexpr const char* kCreateFamilyOnlySymbol = "trtmc_create_family_without_backend";

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context);
extern "C" trtmc::ITask*
trtmc_create_family_without_backend(const trtmc::FamilyOnlyContext& context);
