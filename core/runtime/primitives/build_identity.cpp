/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/plugin_abi.h"

extern "C" const char* trtmc_core_build_id() noexcept {
#ifdef TRTMC_FAKE_INCOMPATIBLE_BUILD
    return "00000000000000000000000000000000";
#else
    return trtmc::kPluginBuildId;
#endif
}
