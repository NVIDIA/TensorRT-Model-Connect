/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dpt/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::dpt {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("dpt engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::dpt

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("dpt does not support --kv-cache-size");
    const auto config_data = trtmc::dpt::require_section(context.reader, "runtime.json");
    auto config =
        trtmc::dpt::parse_runtime_config(std::string(config_data.begin(), config_data.end()));
    const auto plan = trtmc::dpt::require_section(context.reader, "engine.plan");
    auto engine = trtmc::dpt::load_engine(context.backend, plan);
    // The constructor checks the engine's inputs and outputs before first use.
    return new trtmc::dpt::DptPipeline(std::move(engine), std::move(config));
}
