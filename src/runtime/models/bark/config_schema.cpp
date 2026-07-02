/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Registration for the "audio_bark" namespace schema.
// Mirrors python/tensorrt_model_connect/families/bark/runtime_config_schema.py.

#include "runtime/models/bark/config_schema.h"

#include <any>
#include <cstdint>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {
bool is_positive_float(const std::any& value) {
    if (value.type() == typeid(float))
        return std::any_cast<float>(value) > 0.0F;
    if (value.type() == typeid(double))
        return std::any_cast<double>(value) > 0.0;
    return false;
}
} // namespace

Schema make_audio_bark_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "audio_bark",
        {
            ConfigField{"dump_path", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"greedy", "bool", std::any{false}, session, nullptr},
            ConfigField{"seed", "int64", std::any{std::int64_t{-1}}, session, nullptr},
            ConfigField{"fine_temperature", "float", std::any{0.5F}, session, is_positive_float},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_audio_bark_schema, make_audio_bark_schema);
} // namespace trtmc::config::schemas
