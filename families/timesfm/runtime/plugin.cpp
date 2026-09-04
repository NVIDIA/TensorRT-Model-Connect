/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timesfm/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::timesfm {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_number_integer())
        throw std::runtime_error(std::string("TimesFM runtime.json requires integer ") + key);
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error(std::string("TimesFM runtime.json has invalid ") + key);
    return static_cast<std::int32_t>(value);
}

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (!json.is_object() || !json.contains("quantiles") || !json.at("quantiles").is_array() ||
        json.at("quantiles").empty()) {
        throw std::runtime_error("TimesFM runtime.json requires quantiles");
    }
    return RuntimeConfig{require_positive_int(json, "context_length"),
                         require_positive_int(json, "prediction_length"),
                         require_positive_int(json, "frequency_count"),
                         require_positive_int(json, "tensor_parallel_size")};
}

void require_nccl(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size <= 1)
        return;
    static void* const handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("tensor-parallel runtime requires NCCL: " +
                                 std::string(error == nullptr ? "unknown loader error" : error));
    }
}

std::int32_t require_rank(std::int32_t tensor_parallel_size) {
    require_nccl(tensor_parallel_size);
    if (tensor_parallel_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("TimesFM TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tensor_parallel_size)
        throw std::runtime_error("TimesFM RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("TimesFM engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::timesfm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    const auto& runtime = trtmc::timesfm::require_section(context.reader, "runtime.json");
    auto config = trtmc::timesfm::parse_runtime_config(runtime);
    const auto rank = trtmc::timesfm::require_rank(config.tensor_parallel_size);
    const std::string section = config.tensor_parallel_size == 1
                                    ? "engine.plan"
                                    : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& plan = trtmc::timesfm::require_section(context.reader, section.c_str());
    auto engine = trtmc::timesfm::load_engine(context.backend, plan);
    return new trtmc::timesfm::Pipeline(std::move(engine), config);
}
