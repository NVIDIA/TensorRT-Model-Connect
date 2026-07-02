/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Registration for the "platform" namespace schema.
// Mirrors python/tensorrt_model_connect/runtime_config/schemas/platform.py.

#include "trtmc/config/schemas/platform.h"

#include <any>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {
bool is_valid_severity(const std::any& v) {
    if (v.type() != typeid(std::string))
        return false;
    const auto& s = std::any_cast<const std::string&>(v);
    return s == "INTERNAL_ERROR" || s == "ERROR" || s == "WARNING" || s == "INFO" || s == "VERBOSE";
}
} // namespace

Schema make_platform_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "platform",
        {
            ConfigField{"source_dir", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"trt_log_stderr", "bool", std::any{false}, session, nullptr},
            ConfigField{"trt_log_min_severity", "string", std::any{std::string{"INFO"}}, session,
                        is_valid_severity},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_platform_schema, make_platform_schema);
} // namespace trtmc::config::schemas
