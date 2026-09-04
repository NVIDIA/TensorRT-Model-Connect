/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/wan2_2_ti2v/runtime/easycache.h"

#include <string>

namespace trtmc::wan2_2_ti2v {

struct RuntimeConfig {
    EasyCacheConfig easycache;
    bool late_cfg_enabled{false};
};

RuntimeConfig parse_runtime_config(const std::string& json);

} // namespace trtmc::wan2_2_ti2v
