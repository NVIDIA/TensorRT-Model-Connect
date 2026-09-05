/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_vgg/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::timm_vgg {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

TimmVggPreprocessConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    TimmVggPreprocessConfig config;
    config.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.crop_pct = json.at("crop_pct").get<float>();
    config.interpolation = json.at("interpolation").get<std::string>();
    config.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.image_std = json.at("image_std").get<std::vector<float>>();
    if (config.input_image_h <= 0 || config.input_image_w <= 0 || config.crop_pct <= 0.0F ||
        config.crop_pct > 1.0F || config.image_mean.size() != 3 || config.image_std.size() != 3 ||
        (config.interpolation != "bilinear" && config.interpolation != "bicubic")) {
        throw std::runtime_error("timm VGG runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("timm VGG engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::timm_vgg

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("timm_vgg does not support --kv-cache-size");
    const auto& config_data = trtmc::timm_vgg::require_section(context.reader, "runtime.json");
    const auto& plan = trtmc::timm_vgg::require_section(context.reader, "engine.plan");
    auto config = trtmc::timm_vgg::parse_config(config_data);
    auto engine = trtmc::timm_vgg::load_engine(context.backend, plan);
    return new trtmc::TimmVggImageClassificationPipeline(std::move(engine), std::move(config));
}
