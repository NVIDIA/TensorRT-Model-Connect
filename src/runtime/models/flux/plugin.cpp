/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FluxPlugin: handles "diffusion_flux" strategy.
// FLUX diffusion pipeline with T5 + CLIP text encoders, denoiser, and VAE.

#include "diffusion_helpers.h"
#include "plugin_helpers.h"
#include "runtime/models/flux/pipeline.h"
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
    cfg.enabled
        = ((mode == "tensor_parallel" || cfg.context_parallel) && cfg.world_size > 1);
    return cfg;
}

std::string distributed_denoiser_section_name(
    int32_t rank, bool context_parallel) {
    if (context_parallel)
        return "denoiser_plan_cp";
    return "denoiser_plan_tp_rank" + std::to_string(rank);
}

} // namespace

class FluxPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto distributed_config = parse_distributed_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        std::string denoiser_section_name = "denoiser_plan";
        ModuleCreateOptions denoiser_opts = opts;
        const ModuleCreateOptions* denoiser_options = nullptr;
        if (distributed_config.enabled) {
            tp_group = initialize_tensor_parallel_group(distributed_config.world_size);
            denoiser_section_name = distributed_denoiser_section_name(
                tp_group.rank, distributed_config.context_parallel);
            denoiser_opts.distributed_communicator = tp_group.communicator;
            denoiser_opts.distributed_owner = tp_group.owner;
            denoiser_options = &denoiser_opts;
        }

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts,
                                          denoiser_section_name, denoiser_options);

        // Move text encoder modules into vector
        std::vector<std::unique_ptr<TrtModule>> te_modules;
        for (auto& te : parts.text_encoders)
            te_modules.push_back(std::move(te.module));

        // Create native BPE CLIP tokenizer from bundle
        auto clip_tok = create_clip_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<FluxPipeline>(
            std::move(te_modules), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            std::move(clip_tok), ctx.bundle.info.model_id, tp_group.owner, tp_group.rank,
            tp_group.tp_size);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_flux_plugin, FluxPlugin, "diffusion_flux");

} // namespace trtmc
