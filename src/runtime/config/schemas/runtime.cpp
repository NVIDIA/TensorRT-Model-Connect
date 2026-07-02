/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Registration for the "runtime" namespace schema.
// Mirrors python/tensorrt_model_connect/runtime_config/schemas/runtime.py.

#include "trtmc/config/schemas/runtime.h"

#include <any>
#include <set>

namespace trtmc::config::schemas {

Schema make_runtime_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "runtime",
        {
            ConfigField{"disable_cuda_graph", "bool", std::any{false}, session, nullptr},
            ConfigField{"prefer_gpu_greedy", "bool", std::any{false}, session, nullptr},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_runtime_schema, make_runtime_schema);
} // namespace trtmc::config::schemas
