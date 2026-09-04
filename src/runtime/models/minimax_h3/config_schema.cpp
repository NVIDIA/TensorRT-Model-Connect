/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/config_schema.h"

#include <any>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>

namespace trtmc::config::schemas {
namespace {

bool is_positive_finite_double(const std::any& value) {
    if (value.type() != typeid(double))
        return false;
    const double parsed = std::any_cast<double>(value);
    return std::isfinite(parsed) && parsed > 0.0;
}

bool is_positive_budget_gib(const std::any& value) {
    if (value.type() != typeid(std::int64_t))
        return false;
    const auto parsed = std::any_cast<std::int64_t>(value);
    return parsed > 0 && parsed <= (std::numeric_limits<std::int64_t>::max() >> 30);
}

} // namespace

Schema make_minimax_h3_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "minimax_h3",
        {
            ConfigField{"first_block_cache_threshold", "double", std::any{0.08}, session,
                        is_positive_finite_double},
            ConfigField{"retain_engines", "bool", std::any{false}, session, nullptr},
            ConfigField{"retained_tail_weight_budget_gib", "int64", std::any{std::int64_t{24}},
                        session, is_positive_budget_gib},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_minimax_h3_schema, make_minimax_h3_schema);

} // namespace trtmc::config::schemas
