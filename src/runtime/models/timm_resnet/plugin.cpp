/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TimmResnetPlugin: handles timm ResNet image-classification bundles.

#include "plugin_helpers.h"
#include "runtime/models/timm_resnet/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <string>
#include <utility>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

TimmResnetPreprocessConfig make_timm_resnet_preprocess_config(const std::string& json) {
    TimmResnetPreprocessConfig cfg;
    cfg.input_image_h = extract_json_int(json, "input_image_h", cfg.input_image_h);
    cfg.input_image_w = extract_json_int(json, "input_image_w", cfg.input_image_w);
    cfg.crop_pct = extract_json_float(json, "crop_pct", cfg.crop_pct);
    cfg.interpolation = extract_json_string(json, "interpolation", cfg.interpolation);

    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);
    return cfg;
}

} // namespace

class TimmResnetPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        std::string engine_section = "engine_plan";
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            opts.distributed_communicator = tp_group.communicator;
            opts.distributed_owner = tp_group.owner;
            engine_section = tp_engine_section_name(tp_group.rank);
        }

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), "engine_plan", opts);
        return std::make_unique<TimmResnetImageClassificationPipeline>(
            std::move(loaded.module), make_timm_resnet_preprocess_config(ctx.config_json),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_timm_resnet_plugin, TimmResnetPlugin,
                                       "timm_resnet_image_classification");

} // namespace trtmc
