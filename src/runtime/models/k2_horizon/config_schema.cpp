/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/k2_horizon/config_schema.h"

#include <any>
#include <set>

namespace trtmc::config::schemas {

Schema make_k2_horizon_schema() {
    const std::set<Layer> session_only = {Layer::SessionRequest};
    return Schema{
        "k2_horizon",
        {
            ConfigField{"emit_prompt_token_ids", "bool", std::any{false}, session_only, nullptr},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_k2_horizon_schema, make_k2_horizon_schema);

} // namespace trtmc::config::schemas
