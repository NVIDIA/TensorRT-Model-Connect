/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/uni3d/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::uni3d {
namespace {

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("Uni3D bundle is missing " + std::string(name));
    return reader.read_section(name);
}

struct RuntimeConfig {
    std::int32_t num_points;
    std::int32_t min_points;
    std::int32_t input_dim;
    std::int32_t embed_dim;
};

RuntimeConfig parse_runtime_config(const BundleReader& reader) {
    const auto bytes = require_section(reader, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(std::string(bytes.begin(), bytes.end()));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Uni3D invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object() || json.size() != 4)
        throw std::runtime_error("Uni3D runtime.json has an unexpected field set");
    try {
        return {json.at("num_points").get<std::int32_t>(),
                json.at("min_points").get<std::int32_t>(), json.at("input_dim").get<std::int32_t>(),
                json.at("embed_dim").get<std::int32_t>()};
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Uni3D runtime.json is missing an integer field: " +
                                 std::string(error.what()));
    }
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    const auto plan = require_section(context.reader, "engine.plan");
    ModuleCreateOptions options{};
    auto module = context.backend.create_module(plan.data(), plan.size(), options);
    if (module == nullptr || !module->ok())
        throw std::runtime_error("Uni3D failed to load engine.plan");
    return new Uni3DPipeline(std::move(module), config.min_points, config.num_points,
                             config.input_dim, config.embed_dim);
}

} // namespace trtmc::uni3d

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("uni3d does not support --kv-cache-size");
    return trtmc::uni3d::create(context);
}
