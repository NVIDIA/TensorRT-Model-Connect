/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {

Dinov3PreprocessConfig make_preprocess_config(const std::string& json) {
    Dinov3PreprocessConfig config;
    const int32_t image_size = extract_json_int_or_first_array(json, "image_size", 224);
    config.input_image_h = extract_json_int(json, "input_image_h", image_size);
    config.input_image_w = extract_json_int(json, "input_image_w", image_size);
    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        config.image_mean = std::move(mean);
    auto std = extract_json_float_array(json, "image_std", 3);
    if (std.size() == 3)
        config.image_std = std::move(std);
    return config;
}

std::unique_ptr<ITrtModule> load_engine_plan(const PipelineContext& context) {
    const auto* plan = find_section(context.bundle, "engine_plan");
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("Bundle missing engine_plan");
    if (context.backend == nullptr)
        throw std::runtime_error("No backend loaded");

    ModuleCreateOptions options;
    options.runtime_cache_path = context.runtime_cache_path.c_str();
    options.cuda_graphs = context.cuda_graphs;
    const auto start = std::chrono::steady_clock::now();
    auto model = context.backend->create_module(plan->data(), plan->size(), options);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << std::fixed << std::setprecision(6)
              << "[trtmc.load_timing] label=\"engine_plan\" load_deserialize_ms=" << elapsed
              << " plan_bytes=" << plan->size() << '\n';
    if (!model || !model->ok())
        throw std::runtime_error("Failed to create ITrtModule for engine_plan");
    model->set_timing_label("engine_plan");
    return model;
}

} // namespace

class Dinov3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        return std::make_unique<Dinov3ImageFeaturePipeline>(
            load_engine_plan(context), make_preprocess_config(context.config_json),
            context.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_dinov3_plugin, Dinov3Plugin,
                                       "dinov3_image_feature_extraction");

} // namespace trtmc
