/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// EagleVLMPlugin: Eagle VLM-owned embedding and reranking runtime strategies.

#include "plugin_helpers.h"
#include "runtime/models/eagle_vlm/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <string>

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

} // namespace

class EncoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

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
            ctx.backend, find_section(ctx.bundle, engine_section), engine_section.c_str(), opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        const std::string mode =
            (ctx.config.runtime_strategy == "eagle_vlm_reranking") ? "reranking" : "embedding";
        const auto reranking_pooling = extract_json_string(ctx.config_json, "pooling", "last");
        return std::make_unique<EncoderPipeline>(std::move(loaded.module), mode,
                                                 std::move(tokenizer), ctx.bundle.info.model_id,
                                                 reranking_pooling);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_eagle_vlm_plugin, EncoderPlugin,
                                       "eagle_vlm_embedding", "eagle_vlm_reranking");

} // namespace trtmc
