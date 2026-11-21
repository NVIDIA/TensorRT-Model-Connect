/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pointnet/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::pointnet {
namespace {

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("PointNet bundle is missing " + std::string(name));
    return reader.read_section(name);
}

std::string require_text_section(const BundleReader& reader, const char* name) {
    const auto bytes = require_section(reader, name);
    return std::string(bytes.begin(), bytes.end());
}

struct RuntimeConfig {
    std::int32_t num_points;
    std::int32_t num_classes;
    std::int32_t input_dim;
};

RuntimeConfig parse_runtime_config(const BundleReader& reader) {
    const std::string text = require_text_section(reader, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("PointNet invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object() || json.size() != 3)
        throw std::runtime_error("PointNet runtime.json has an unexpected field set");
    RuntimeConfig config{};
    try {
        config.num_points = json.at("num_points").get<std::int32_t>();
        config.num_classes = json.at("num_classes").get<std::int32_t>();
        config.input_dim = json.at("input_dim").get<std::int32_t>();
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("PointNet runtime.json is missing an integer field: " +
                                 std::string(error.what()));
    }
    if (config.num_points <= 0 || config.num_classes <= 0 || config.input_dim <= 0)
        throw std::runtime_error("PointNet runtime.json dimensions must be positive");
    return config;
}

std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan,
                                        cudaStream_t stream) {
    ModuleCreateOptions options{};
    options.stream = stream;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (module == nullptr || !module->ok())
        throw std::runtime_error("PointNet failed to load engine.plan");
    return module;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    const auto plan = require_section(context.reader, "engine.plan");
    auto module = load_module(context.backend, plan, nullptr);
    return new PointNetPipeline(std::move(module), config.num_points, config.num_classes,
                                config.input_dim);
}

} // namespace trtmc::pointnet

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("pointnet does not support --kv-cache-size");
    return trtmc::pointnet::create(context);
}
