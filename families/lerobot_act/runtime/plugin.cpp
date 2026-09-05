/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_act/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::lerobot_act {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("LeRobot ACT bundle is missing " + std::string(name));
    return bundle.read_section(name);
}

struct RuntimeConfig {
    std::int32_t image_height;
    std::int32_t image_width;
    std::int32_t image_channels;
    std::int32_t state_dim;
    std::int32_t action_dim;
    std::int32_t chunk_size;
    std::vector<float> action_min;
    std::vector<float> action_max;
};

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    const std::vector<std::string> fields{
        "image_height", "image_width", "image_channels", "state_dim",
        "action_dim",   "chunk_size",  "action_min",     "action_max",
    };
    if (!json.is_object() || json.size() != fields.size())
        throw std::runtime_error("LeRobot ACT runtime.json has an unexpected field set");
    for (const auto& field : fields) {
        if (!json.contains(field))
            throw std::runtime_error("LeRobot ACT runtime.json is missing " + field);
    }
    RuntimeConfig config{
        json.at("image_height").get<std::int32_t>(),
        json.at("image_width").get<std::int32_t>(),
        json.at("image_channels").get<std::int32_t>(),
        json.at("state_dim").get<std::int32_t>(),
        json.at("action_dim").get<std::int32_t>(),
        json.at("chunk_size").get<std::int32_t>(),
        json.at("action_min").get<std::vector<float>>(),
        json.at("action_max").get<std::vector<float>>(),
    };
    if (config.image_height <= 0 || config.image_width <= 0 || config.image_channels != 3 ||
        config.state_dim <= 0 || config.action_dim <= 0 || config.chunk_size <= 0 ||
        config.action_min.size() != static_cast<std::size_t>(config.action_dim) ||
        config.action_max.size() != static_cast<std::size_t>(config.action_dim)) {
        throw std::runtime_error("LeRobot ACT runtime.json has invalid dimensions");
    }
    return config;
}

} // namespace

ITask* create(const FamilyContext& context) {
    auto config = parse_config(require_section(context.reader, "runtime.json"));
    const auto plan = require_section(context.reader, "engine.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (engine == nullptr || !engine->ok())
        throw std::runtime_error("LeRobot ACT failed to load engine.plan");
    engine->set_timing_label("LeRobot ACT policy");
    return new Pipeline(std::move(engine), config.image_height, config.image_width,
                        config.image_channels, config.state_dim, config.action_dim,
                        config.chunk_size, std::move(config.action_min),
                        std::move(config.action_max));
}

} // namespace trtmc::lerobot_act

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lerobot_act does not support --kv-cache-size");
    return trtmc::lerobot_act::create(context);
}
