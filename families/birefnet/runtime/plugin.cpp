/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/birefnet/runtime/pipeline.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::birefnet {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("birefnet bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

BiRefNetRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    BiRefNetRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.preprocess.image_std = json.at("image_std").get<std::vector<float>>();
    config.mask_threshold = json.at("mask_threshold").get<float>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.preprocess.image_mean.size() != 3 || config.preprocess.image_std.size() != 3) {
        throw std::runtime_error("birefnet runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("birefnet engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::birefnet

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("birefnet does not support --kv-cache-size");
    namespace bn = trtmc::birefnet;
    const auto config_data = bn::require_section(context.reader, "runtime.json");
    const auto plan = bn::require_section(context.reader, "segmenter.plan");
    auto config = bn::parse_config(config_data);
    auto engine = bn::load_engine(context.backend, plan);
    return new trtmc::BiRefNetSegmentationPipeline(std::move(engine), std::move(config));
}
