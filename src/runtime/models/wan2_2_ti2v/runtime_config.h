/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/wan2_2_ti2v/easycache.h"

namespace trtmc::config {
class ConfigBundle;
}

namespace trtmc::wan2_2_ti2v {

struct RuntimeConfig {
    EasyCacheConfig easycache;
    bool late_cfg_enabled{false};
};

// Copy the immutable session config while PipelineContext is alive. The
// pipeline owns the returned values and never retains ConfigBundle pointers.
RuntimeConfig resolve_runtime_config(const config::ConfigBundle* config);

} // namespace trtmc::wan2_2_ti2v
