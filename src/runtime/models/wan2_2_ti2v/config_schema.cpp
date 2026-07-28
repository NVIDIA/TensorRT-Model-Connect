/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/config_schema.h"

#include <any>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>

namespace trtmc::config::schemas {
namespace {

bool is_positive_double(const std::any& value) {
    if (value.type() != typeid(double))
        return false;
    const double parsed = std::any_cast<double>(value);
    return std::isfinite(parsed) && parsed > 0.0;
}

bool is_nonnegative_int32(const std::any& value) {
    if (value.type() != typeid(std::int64_t))
        return false;
    const auto parsed = std::any_cast<std::int64_t>(value);
    return parsed >= 0 && parsed <= std::numeric_limits<std::int32_t>::max();
}

bool is_positive_int32(const std::any& value) {
    if (!is_nonnegative_int32(value))
        return false;
    return std::any_cast<std::int64_t>(value) > 0;
}

} // namespace

Schema make_wan2_2_ti2v_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "wan2_2_ti2v",
        {
            ConfigField{"easycache_enabled", "bool", std::any{false}, session, nullptr},
            ConfigField{"easycache_threshold", "double", std::any{0.02}, session,
                        is_positive_double},
            ConfigField{"easycache_first_exact_steps", "int64", std::any{std::int64_t{7}}, session,
                        is_nonnegative_int32},
            ConfigField{"easycache_last_exact_steps", "int64", std::any{std::int64_t{2}}, session,
                        is_nonnegative_int32},
            ConfigField{"easycache_max_consecutive_reuse", "int64", std::any{std::int64_t{1}},
                        session, is_positive_int32},
            ConfigField{"late_cfg_enabled", "bool", std::any{false}, session, nullptr},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_wan2_2_ti2v_schema, make_wan2_2_ti2v_schema);

} // namespace trtmc::config::schemas
