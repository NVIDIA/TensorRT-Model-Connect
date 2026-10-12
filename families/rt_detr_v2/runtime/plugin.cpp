/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rt_detr_v2/runtime/pipeline.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::rt_detr_v2 {
namespace {

// read_section returns by value, so this does too: binding the result to a
// reference that outlives the call would dangle.
std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("rt_detr_v2 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

RtDetrRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RtDetrRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.num_queries = json.at("num_queries").get<std::int32_t>();
    config.num_labels = json.at("num_labels").get<std::int32_t>();
    config.score_threshold = json.at("score_threshold").get<float>();
    if (json.value("do_normalize", false)) {
        // The builder writes this false on purpose; a true here would mean the
        // bundle disagrees with the checkpoint's own preprocessor.
        throw std::runtime_error("rt_detr_v2 does not normalise its input");
    }
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.num_queries <= 0 || config.num_labels <= 0) {
        throw std::runtime_error("rt_detr_v2 runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("rt_detr_v2 engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::rt_detr_v2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("rt_detr_v2 does not support --kv-cache-size");
    namespace rt = trtmc::rt_detr_v2;
    const auto config_data = rt::require_section(context.reader, "runtime.json");
    const auto plan = rt::require_section(context.reader, "detector.plan");
    auto config = rt::parse_config(config_data);
    auto engine = rt::load_engine(context.backend, plan);
    return new trtmc::RtDetrV2ObjectDetectionPipeline(std::move(engine), std::move(config));
}
