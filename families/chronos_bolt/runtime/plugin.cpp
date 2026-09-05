/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chronos_bolt/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::chronos_bolt {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_number_integer())
        throw std::runtime_error(std::string("Chronos-Bolt runtime.json requires integer ") + key);
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(std::string("Chronos-Bolt runtime.json has invalid ") + key);
    return static_cast<std::int32_t>(value);
}

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (!json.is_object() || !json.contains("quantiles") || !json.at("quantiles").is_array() ||
        json.at("quantiles").empty()) {
        throw std::runtime_error("Chronos-Bolt runtime.json requires a nonempty quantiles array");
    }
    return RuntimeConfig{require_positive_int(json, "context_length"),
                         require_positive_int(json, "prediction_length"),
                         static_cast<std::int32_t>(json.at("quantiles").size()),
                         require_positive_int(json, "tensor_parallel_size")};
}

std::int32_t require_rank(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("Chronos-Bolt TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tensor_parallel_size)
        throw std::runtime_error("Chronos-Bolt RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("Chronos-Bolt engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::chronos_bolt

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("chronos_bolt does not support --kv-cache-size");
    const auto& runtime = trtmc::chronos_bolt::require_section(context.reader, "runtime.json");
    auto config = trtmc::chronos_bolt::parse_runtime_config(runtime);
    const auto rank = trtmc::chronos_bolt::require_rank(config.tensor_parallel_size);
    const std::string section = config.tensor_parallel_size == 1
                                    ? "engine.plan"
                                    : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& plan = trtmc::chronos_bolt::require_section(context.reader, section.c_str());
    auto engine = trtmc::chronos_bolt::load_engine(context.backend, plan);
    return new trtmc::chronos_bolt::Pipeline(std::move(engine), config);
}
