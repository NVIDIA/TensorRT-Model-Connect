/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SegformerPlugin: SegFormer-owned semantic segmentation strategy.

#include "plugin_helpers.h"
#include "segment_pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <memory>
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

SegformerPreprocessConfig make_segformer_preprocess_config(const std::string& json) {
    SegformerPreprocessConfig cfg;
    cfg.num_classes = extract_json_int(json, "num_classes", cfg.num_classes);
    cfg.input_image_h = extract_json_int(json, "input_image_h", cfg.input_image_h);
    cfg.input_image_w = extract_json_int(json, "input_image_w", cfg.input_image_w);
    cfg.output_h = extract_json_int(json, "output_h", cfg.output_h);
    cfg.output_w = extract_json_int(json, "output_w", cfg.output_w);

    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);
    return cfg;
}

} // namespace

class SegformerPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        ModuleCreateOptions encoder_opts = opts;
        std::string encoder_section = "engine_plan";
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            encoder_opts.distributed_communicator = tp_group.communicator;
            encoder_opts.distributed_owner = tp_group.owner;
            encoder_section = tp_engine_section_name(tp_group.rank);
        }

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, encoder_section), "engine_plan", encoder_opts);
        return std::make_unique<SegmentPipeline>(std::move(loaded.module),
                                                 make_segformer_preprocess_config(ctx.config_json),
                                                 ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_segformer_plugin, SegformerPlugin,
                                       "segformer_segmentation");

} // namespace trtmc
