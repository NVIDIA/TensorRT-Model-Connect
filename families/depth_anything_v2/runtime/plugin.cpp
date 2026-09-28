/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything_v2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::depth_anything_v2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

DepthAnythingV2PreprocessConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    DepthAnythingV2PreprocessConfig config;
    config.input_image_size = json.at("input_image_size").get<std::int32_t>();
    config.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.image_std = json.at("image_std").get<std::vector<float>>();
    if (config.input_image_size <= 0 || config.image_mean.size() != 3 ||
        config.image_std.size() != 3) {
        throw std::runtime_error(
            "Depth Anything V2 runtime.json does not match its preprocessing contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("Depth Anything V2 engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::depth_anything_v2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("depth_anything_v2 does not support --kv-cache-size");
    const auto& config_data =
        trtmc::depth_anything_v2::require_section(context.reader, "runtime.json");
    const auto& plan = trtmc::depth_anything_v2::require_section(context.reader, "engine.plan");
    auto config = trtmc::depth_anything_v2::parse_config(config_data);
    auto engine = trtmc::depth_anything_v2::load_engine(context.backend, plan);
    return new trtmc::DepthAnythingV2Pipeline(std::move(engine), std::move(config));
}
