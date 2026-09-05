/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtst/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::patchtst {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_number_integer())
        throw std::runtime_error(std::string("PatchTST runtime.json requires integer ") + key);
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(std::string("PatchTST runtime.json has invalid ") + key);
    return static_cast<std::int32_t>(value);
}

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (!json.is_object())
        throw std::runtime_error("PatchTST runtime.json must be an object");
    const auto task = json.at("task").get<std::string>();
    std::string output_name;
    if (task == "forecast")
        output_name = "prediction_outputs";
    else if (task == "regression" || task == "classification")
        output_name = "regression_outputs";
    else
        throw std::runtime_error("PatchTST runtime.json has unsupported task: " + task);
    return RuntimeConfig{require_positive_int(json, "context_length"),
                         require_positive_int(json, "num_input_channels"),
                         require_positive_int(json, "prediction_length"),
                         require_positive_int(json, "tensor_parallel_size"),
                         std::move(output_name)};
}

std::int32_t require_rank(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("PatchTST TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tensor_parallel_size)
        throw std::runtime_error("PatchTST RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("PatchTST engine failed to load");
    return module;
}

} // namespace
} // namespace trtmc::patchtst

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("patchtst does not support --kv-cache-size");
    const auto& runtime = trtmc::patchtst::require_section(context.reader, "runtime.json");
    auto config = trtmc::patchtst::parse_runtime_config(runtime);
    const auto rank = trtmc::patchtst::require_rank(config.tensor_parallel_size);
    const std::string section = config.tensor_parallel_size == 1
                                    ? "engine.plan"
                                    : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& plan = trtmc::patchtst::require_section(context.reader, section.c_str());
    auto module = trtmc::patchtst::load_module(context.backend, plan);
    return new trtmc::patchtst::Pipeline(std::move(module), std::move(config));
}
