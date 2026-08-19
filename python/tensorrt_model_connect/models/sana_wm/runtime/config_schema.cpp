/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config_schema.h"

#include <any>
#include <cstdint>
#include <set>
#include <string>

namespace trtmc::config::schemas {

Schema make_sana_wm_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "sana_wm",
        {
            ConfigField{"image_path", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"action", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"translation_speed", "float", std::any{-1.0F}, session, nullptr},
            ConfigField{"rotation_speed_deg", "float", std::any{-1.0F}, session, nullptr},
            ConfigField{"num_frames", "int64", std::any{std::int64_t{-1}}, session, nullptr},
            ConfigField{"fps", "int64", std::any{std::int64_t{-1}}, session, nullptr},
            ConfigField{"flow_shift", "float", std::any{-1.0F}, session, nullptr},
            ConfigField{"intrinsics", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"no_refiner", "bool", std::any{false}, session, nullptr},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_sana_wm_schema, make_sana_wm_schema);

} // namespace trtmc::config::schemas
