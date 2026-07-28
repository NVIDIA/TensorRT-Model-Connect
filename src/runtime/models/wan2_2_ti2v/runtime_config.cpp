/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/runtime_config.h"

#include "trtmc/config/config_bundle.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr const char* kNamespace = "wan2_2_ti2v";

int32_t checked_int32(std::int64_t value, const char* field) {
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::out_of_range(std::string(kNamespace) + "." + field +
                                " is outside the non-negative int32 range");
    }
    return static_cast<int32_t>(value);
}

} // namespace

RuntimeConfig resolve_runtime_config(const config::ConfigBundle* config) {
    RuntimeConfig result;
    if (config == nullptr)
        return result;

    try {
        result.easycache.enabled = config->get<bool>(kNamespace, "easycache_enabled");
        result.easycache.threshold = config->get<double>(kNamespace, "easycache_threshold");
        result.easycache.first_exact_steps =
            checked_int32(config->get<std::int64_t>(kNamespace, "easycache_first_exact_steps"),
                          "easycache_first_exact_steps");
        result.easycache.last_exact_steps =
            checked_int32(config->get<std::int64_t>(kNamespace, "easycache_last_exact_steps"),
                          "easycache_last_exact_steps");
        result.easycache.max_consecutive_reuse =
            checked_int32(config->get<std::int64_t>(kNamespace, "easycache_max_consecutive_reuse"),
                          "easycache_max_consecutive_reuse");
        result.late_cfg_enabled = config->get<bool>(kNamespace, "late_cfg_enabled");
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("Invalid Wan2.2 runtime config: ") + error.what());
    }
    return result;
}

} // namespace trtmc::wan2_2_ti2v
