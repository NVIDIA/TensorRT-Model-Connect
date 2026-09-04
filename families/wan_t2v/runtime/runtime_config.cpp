/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan_t2v/runtime/runtime_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::wan_t2v {
namespace {

bool supported_distributed_size(std::int32_t size) {
    return size == 2 || size == 4 || size == 8;
}

} // namespace

ParallelRuntimeConfig parse_parallel_runtime_config(const std::string& json) {
    const auto document = nlohmann::json::parse(json);
    if (!document.contains("parallel_mode") || !document.at("parallel_mode").is_string())
        throw std::runtime_error("Wan runtime.json requires string parallel_mode");
    if (!document.contains("parallel_size") || !document.at("parallel_size").is_number_integer())
        throw std::runtime_error("Wan runtime.json requires integer parallel_size");

    ParallelRuntimeConfig config;
    const auto mode = document.at("parallel_mode").get<std::string>();
    if (mode == "single")
        config.mode = ParallelMode::Single;
    else if (mode == "tensor_parallel")
        config.mode = ParallelMode::Tensor;
    else if (mode == "context_parallel")
        config.mode = ParallelMode::Context;
    else
        throw std::runtime_error("Wan runtime.json has unsupported parallel_mode");
    config.size = document.at("parallel_size").get<std::int32_t>();

    if ((config.mode == ParallelMode::Single && config.size != 1) ||
        (config.mode != ParallelMode::Single && !supported_distributed_size(config.size))) {
        throw std::runtime_error("Wan runtime.json has invalid parallel settings");
    }
    return config;
}

std::string denoiser_section_name(const ParallelRuntimeConfig& config, std::int32_t rank) {
    if (rank < 0 || rank >= config.size)
        throw std::runtime_error("Wan rank is outside parallel_size");
    if (config.mode == ParallelMode::Tensor)
        return "denoiser.rank" + std::to_string(rank) + ".plan";
    return "denoiser.plan";
}

} // namespace trtmc::wan_t2v
