/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolov10/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::yolov10 {
namespace {

// Detections below this are dropped. The head always emits its full slot count,
// padded with low-scoring entries, so something has to end the list.
constexpr float kScoreThreshold = 0.25F;

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

Yolov10PreprocessConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Yolov10PreprocessConfig config;
    config.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.pad_value = json.at("pad_value").get<float>();
    if (config.input_image_h <= 0 || config.input_image_w <= 0 || config.pad_value < 0.0F ||
        config.pad_value > 1.0F) {
        throw std::runtime_error("YOLOv10 runtime.json does not match its contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("YOLOv10 engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::yolov10

TRTMC_DEFINE_FAMILY_PLUGIN_V1("yolov10")

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("yolov10 does not support --kv-cache-size");
    const auto config_data = trtmc::yolov10::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::yolov10::require_section(context.reader, "engine.plan");
    auto config = trtmc::yolov10::parse_config(config_data);
    auto engine = trtmc::yolov10::load_engine(context.backend, plan);
    return new trtmc::Yolov10ObjectDetectionPipeline(std::move(engine), std::move(config),
                                                     trtmc::yolov10::kScoreThreshold);
}
