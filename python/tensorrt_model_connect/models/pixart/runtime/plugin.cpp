/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// PixArtPlugin: handles "diffusion_pixart" strategy.
// PixArt-Sigma/Alpha via TRT Network API.

#include "diffusion_helpers.h"
#include "pipeline.h"
#include "plugin_helpers.h"
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

std::string tp_denoiser_section_name(int32_t rank) {
    return "denoiser_plan_tp_rank" + std::to_string(rank);
}

} // namespace

class PixArtPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        std::string denoiser_section_name = "denoiser_plan";
        ModuleCreateOptions denoiser_opts = opts;
        const ModuleCreateOptions* denoiser_options = nullptr;
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            denoiser_section_name = tp_denoiser_section_name(tp_group.rank);
            denoiser_opts.distributed_communicator = tp_group.communicator;
            denoiser_opts.distributed_owner = tp_group.owner;
            denoiser_options = &denoiser_opts;
        }

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts,
                                          denoiser_section_name, denoiser_options);

        if (ctx.cuda_graphs) {
            parts.denoiser.module->enable_cuda_graph();
            parts.vae.module->enable_cuda_graph();
            for (auto& text_encoder : parts.text_encoders)
                text_encoder.module->enable_cuda_graph();
        }

        // Extract first text encoder
        std::unique_ptr<TrtModule> te_module;
        if (!parts.text_encoders.empty())
            te_module = std::move(parts.text_encoders[0].module);

        return std::make_unique<PixArtPipeline>(
            std::move(te_module), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            ctx.bundle.info.model_id, tp_group.owner, tp_group.rank, tp_group.tp_size);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_pixart_plugin, PixArtPlugin, "diffusion_pixart");

} // namespace trtmc
