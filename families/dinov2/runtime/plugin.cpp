/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::dinov2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

Dinov2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Dinov2RuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<int32_t>();
    config.preprocess.resize_shortest_edge = json.at("resize_shortest_edge").get<int32_t>();
    config.preprocess.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.preprocess.image_std = json.at("image_std").get<std::vector<float>>();
    config.patch_size = json.at("patch_size").get<int32_t>();
    config.hidden_size = json.at("hidden_size").get<int32_t>();
    config.num_register_tokens = json.at("num_register_tokens").get<int32_t>();
    return config;
}

} // namespace
} // namespace trtmc::dinov2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("dinov2 does not support --kv-cache-size");
    const auto config_data = trtmc::dinov2::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::dinov2::require_section(context.reader, "engine.plan");
    auto config = trtmc::dinov2::parse_config(config_data);
    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("DINOv2 engine failed to load");
    return new trtmc::Dinov2FeaturePipeline(std::move(engine), std::move(config));
}
