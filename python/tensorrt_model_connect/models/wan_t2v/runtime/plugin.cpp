/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// WanPlugin: handles "diffusion_wan" strategy only.
// Uses WanPipeline with a single text encoder, denoiser, and VAE.

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

struct DistributedRuntimeConfig {
    bool enabled{false};
    bool context_parallel{false};
    int32_t world_size{1};
};

DistributedRuntimeConfig parse_distributed_runtime_config(const std::string& config_json) {
    DistributedRuntimeConfig cfg;
    auto mode = extract_json_string(config_json, "parallel_mode", "single");
    if (mode == "single")
        mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    if (mode == "single")
        mode = extract_json_string(config_json, "context_parallel_mode", "single");
    cfg.context_parallel = (mode == "context_parallel");
    cfg.world_size = cfg.context_parallel
                         ? extract_json_int(config_json, "context_parallel_size", 1)
                         : extract_json_int(config_json, "tensor_parallel_size", 1);
    cfg.enabled = ((mode == "tensor_parallel" || cfg.context_parallel) && cfg.world_size > 1);
    return cfg;
}

std::string distributed_denoiser_section_name(int32_t rank, bool context_parallel) {
    if (context_parallel)
        return "denoiser_plan_cp";
    return "denoiser_plan_tp_rank" + std::to_string(rank);
}

} // namespace

class WanPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto distributed_config = parse_distributed_runtime_config(ctx.config_json);
        DistributedRuntimeGroup distributed_group;
        std::string denoiser_section_name = "denoiser_plan";
        ModuleCreateOptions denoiser_opts = opts;
        const ModuleCreateOptions* denoiser_options = nullptr;
        if (distributed_config.enabled) {
            distributed_group = initialize_tensor_parallel_group(distributed_config.world_size);
            denoiser_section_name = distributed_denoiser_section_name(
                distributed_group.rank, distributed_config.context_parallel);
            denoiser_opts.distributed_communicator = distributed_group.communicator;
            denoiser_opts.distributed_owner = distributed_group.owner;
            denoiser_options = &denoiser_opts;
        }

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts,
                                          denoiser_section_name, denoiser_options);

        if (ctx.cuda_graphs) {
            parts.denoiser.module->enable_cuda_graph();
            parts.vae.module->enable_cuda_graph();
            if (parts.vae_first_frame.module)
                parts.vae_first_frame.module->enable_cuda_graph();
            for (auto& text_encoder : parts.text_encoders)
                text_encoder.module->enable_cuda_graph();
        }

        // Extract first text encoder
        std::unique_ptr<TrtModule> te_module;
        if (!parts.text_encoders.empty())
            te_module = std::move(parts.text_encoders[0].module);

        return std::make_unique<WanPipeline>(
            std::move(te_module), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            ctx.bundle.info.model_id, distributed_group.owner, distributed_group.rank,
            distributed_group.world_size, std::move(parts.vae_first_frame.module));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_wan_plugin, WanPlugin, "diffusion_wan");

} // namespace trtmc
