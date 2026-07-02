/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Registration for the "text_trace" namespace schema.
//
// Mirrors python/tensorrt_model_connect/runtime_config/schemas/text_trace.py.

#include "trtmc/config/schemas/text_trace.h"

#include <any>
#include <cstdint>
#include <limits>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {
bool is_nonneg_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 0;
}
bool is_positive_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 1;
}
} // namespace

Schema make_text_trace_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "text_trace",
        {
            ConfigField{"step_trace_path", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"step_trace_start_pos", "int32", std::any{std::int32_t{0}}, session,
                        is_nonneg_int32},
            ConfigField{"step_trace_end_pos", "int32", std::any{std::int32_t{2'000'000'000}},
                        session, is_nonneg_int32},
            ConfigField{"step_trace_topk", "int32", std::any{std::int32_t{8}}, session,
                        is_positive_int32},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_text_trace_schema, make_text_trace_schema);
} // namespace trtmc::config::schemas
