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
#include <vector>

namespace trtmc::dinov2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::vector<float> triple(const nlohmann::json& json, const char* name) {
    auto values = json.at(name).get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error(std::string("dinov2 runtime.json has invalid ") + name);
    return values;
}

Dinov2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Dinov2RuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.shortest_edge = json.at("shortest_edge").get<std::int32_t>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.resample = json.at("resample").get<std::int32_t>();
    config.patch_size = json.at("patch_size").get<std::int32_t>();
    config.grid = json.at("grid").get<std::int32_t>();
    config.hidden_size = json.at("hidden_size").get<std::int32_t>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.preprocess.input_image_h != config.preprocess.input_image_w ||
        config.patch_size <= 0 || config.grid <= 0 || config.hidden_size <= 0 ||
        config.preprocess.shortest_edge < config.preprocess.input_image_h) {
        throw std::runtime_error("dinov2 runtime.json does not match its runtime contract");
    }
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImageToTokenAndPooledFeatures::kTask)
        throw std::runtime_error("dinov2 bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::dinov2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("dinov2 does not support --kv-cache-size");
    const auto config_data = trtmc::dinov2::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::dinov2::require_section(context.reader, "model.plan");

    auto config = trtmc::dinov2::parse_config(config_data);
    auto primary_task = trtmc::dinov2::require_task(context.reader);
    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("dinov2 engine failed to load");
    return new trtmc::Dinov2Pipeline(std::move(engine), std::move(config), std::move(primary_task));
}
