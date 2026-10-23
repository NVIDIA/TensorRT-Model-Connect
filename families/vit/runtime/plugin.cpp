/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vit/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::vit {
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
        throw std::runtime_error(std::string("vit runtime.json has invalid ") + name);
    return values;
}

VitRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    VitRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    const auto resample = json.at("resample").get<std::int32_t>();
    if (resample != static_cast<std::int32_t>(VitResample::kBilinear) &&
        resample != static_cast<std::int32_t>(VitResample::kBicubic)) {
        throw std::runtime_error("vit runtime.json names an unsupported resize filter");
    }
    config.preprocess.resample = static_cast<VitResample>(resample);
    config.num_classes = json.at("num_classes").get<std::int32_t>();
    config.labels = json.at("labels").get<std::vector<std::string>>();
    config.vocabulary_id = json.at("vocabulary_id").get<std::string>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.preprocess.input_image_h != config.preprocess.input_image_w ||
        config.num_classes <= 0) {
        throw std::runtime_error("vit runtime.json does not match its runtime contract");
    }
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImageToClassScores::kTask)
        throw std::runtime_error("vit bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::vit

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("vit does not support --kv-cache-size");
    const auto config_data = trtmc::vit::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::vit::require_section(context.reader, "model.plan");

    auto config = trtmc::vit::parse_config(config_data);
    auto primary_task = trtmc::vit::require_task(context.reader);
    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("vit engine failed to load");
    return new trtmc::VitPipeline(std::move(engine), std::move(config), std::move(primary_task));
}
