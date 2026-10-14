/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx2/runtime/runtime_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::ltx2 {
namespace {

bool supported_context_parallel_size(std::int32_t size) {
    return size == 2 || size == 4 || size == 8;
}

} // namespace

ParallelRuntimeConfig parse_parallel_runtime_config(const std::string& json) {
    const auto document = nlohmann::json::parse(json);
    const bool has_mode = document.contains("parallel_mode");
    const bool has_size = document.contains("parallel_size");
    if (!has_mode && !has_size)
        return {};
    if (!has_mode || !document.at("parallel_mode").is_string())
        throw std::runtime_error("LTX-2.5 runtime.json requires string parallel_mode");
    if (!has_size || !document.at("parallel_size").is_number_integer())
        throw std::runtime_error("LTX-2.5 runtime.json requires integer parallel_size");

    ParallelRuntimeConfig config;
    const auto mode = document.at("parallel_mode").get<std::string>();
    if (mode == "single")
        config.mode = ParallelMode::Single;
    else if (mode == "context_parallel")
        config.mode = ParallelMode::Context;
    else
        throw std::runtime_error("LTX-2.5 runtime.json has unsupported parallel_mode");
    config.size = document.at("parallel_size").get<std::int32_t>();

    if ((config.mode == ParallelMode::Single && config.size != 1) ||
        (config.mode == ParallelMode::Context && !supported_context_parallel_size(config.size))) {
        throw std::runtime_error("LTX-2.5 runtime.json has invalid parallel settings");
    }
    return config;
}

} // namespace trtmc::ltx2
