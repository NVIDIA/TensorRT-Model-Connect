/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/task.h"

namespace trtmc {

class IBackend;

struct FamilyContext {
    const BundleReader& reader;
    IBackend& backend;
};

using CreateFamilyFn = ITask* (*)(const FamilyContext& context);

inline constexpr const char* kCreateFamilySymbol = "trtmc_create_family";

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context);
