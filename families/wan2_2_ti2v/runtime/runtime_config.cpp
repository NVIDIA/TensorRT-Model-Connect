/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan2_2_ti2v/runtime/runtime_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::wan2_2_ti2v {

RuntimeConfig parse_runtime_config(const std::string& text) {
    const auto json = nlohmann::json::parse(text);
    RuntimeConfig config;
    config.easycache.enabled = json.at("easycache_enabled").get<bool>();
    config.easycache.threshold = json.at("easycache_threshold").get<double>();
    config.easycache.first_exact_steps = json.at("easycache_first_exact_steps").get<std::int32_t>();
    config.easycache.last_exact_steps = json.at("easycache_last_exact_steps").get<std::int32_t>();
    config.easycache.max_consecutive_reuse =
        json.at("easycache_max_consecutive_reuse").get<std::int32_t>();
    config.late_cfg_enabled = json.at("late_cfg_enabled").get<bool>();
    if (config.easycache.threshold < 0.0 || config.easycache.first_exact_steps < 0 ||
        config.easycache.last_exact_steps < 0 || config.easycache.max_consecutive_reuse < 0)
        throw std::runtime_error("Wan2.2 runtime cache configuration is invalid");
    return config;
}

} // namespace trtmc::wan2_2_ti2v
