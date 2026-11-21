/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/oneformer/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::oneformer {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("oneformer bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RuntimeConfig config;
    config.shortest_edge = json.at("shortest_edge").get<std::int32_t>();
    config.longest_edge = json.at("longest_edge").get<std::int32_t>();
    const auto mean = json.at("image_mean").get<std::vector<float>>();
    const auto deviation = json.at("image_std").get<std::vector<float>>();
    config.normalization.rescale_factor = json.at("rescale_factor").get<double>();
    config.class_names = json.at("class_names").get<std::vector<std::string>>();
    if (config.shortest_edge <= 0 || config.longest_edge <= config.shortest_edge ||
        mean.size() != 3 || deviation.size() != 3 || !(config.normalization.rescale_factor > 0.0) ||
        config.class_names.empty())
        throw std::runtime_error("oneformer runtime.json does not match its runtime contract");
    for (std::size_t channel = 0; channel < 3; ++channel) {
        if (!(deviation[channel] > 0.0F))
            throw std::runtime_error("oneformer image_std must be positive");
        config.normalization.mean[channel] = mean[channel];
        config.normalization.std[channel] = deviation[channel];
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("oneformer engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::oneformer

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("oneformer does not support --kv-cache-size");
    auto config = trtmc::oneformer::parse_config(
        trtmc::oneformer::require_section(context.reader, "runtime.json"));
    auto engine = trtmc::oneformer::load_engine(
        context.backend, trtmc::oneformer::require_section(context.reader, "engine.plan"));
    return new trtmc::oneformer::SemanticSegmentation(std::move(engine), std::move(config));
}
