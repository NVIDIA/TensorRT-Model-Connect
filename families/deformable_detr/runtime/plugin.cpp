/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deformable_detr/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::deformable_detr {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::array<float, 3> read_triple(const nlohmann::json& json, const char* name,
                                 const std::array<float, 3>& fallback) {
    if (!json.contains(name))
        return fallback;
    const auto values = json.at(name).get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error("deformable_detr " + std::string(name) + " must have 3 entries");
    return {values[0], values[1], values[2]};
}

DeformableDetrRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    DeformableDetrRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.do_normalize = json.at("do_normalize").get<bool>();
    config.preprocess.image_mean = read_triple(json, "image_mean", config.preprocess.image_mean);
    config.preprocess.image_std = read_triple(json, "image_std", config.preprocess.image_std);
    config.num_queries = json.at("num_queries").get<std::int32_t>();
    config.num_labels = json.at("num_labels").get<std::int32_t>();
    config.score_threshold = json.at("score_threshold").get<float>();
    if (json.contains("top_k"))
        config.top_k = json.at("top_k").get<std::int32_t>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.num_queries <= 0 || config.num_labels <= 0 || config.top_k <= 0) {
        throw std::runtime_error("deformable_detr runtime.json does not match its contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("deformable_detr engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::deformable_detr

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("deformable_detr does not support --kv-cache-size");
    namespace dd = trtmc::deformable_detr;
    const auto config_data = dd::require_section(context.reader, "runtime.json");
    const auto plan = dd::require_section(context.reader, "detector.plan");
    auto config = dd::parse_config(config_data);
    auto engine = dd::load_engine(context.backend, plan);
    return new trtmc::DeformableDetrObjectDetectionPipeline(std::move(engine), std::move(config));
}
