/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/config_schema.h"

#include <any>
#include <cmath>
#include <set>

namespace trtmc::config::schemas {
namespace {

bool is_positive_finite_double(const std::any& value) {
    if (value.type() != typeid(double))
        return false;
    const double parsed = std::any_cast<double>(value);
    return std::isfinite(parsed) && parsed > 0.0;
}

} // namespace

Schema make_minimax_h3_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "minimax_h3",
        {
            ConfigField{"first_block_cache_threshold", "double", std::any{0.08}, session,
                        is_positive_finite_double},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_minimax_h3_schema, make_minimax_h3_schema);

} // namespace trtmc::config::schemas
