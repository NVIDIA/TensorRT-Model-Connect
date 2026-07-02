/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MambaPlugin: handles the Mamba-owned recurrent strategy.
// Mamba/SSM models with conv_state + ssm_state recurrent state.

#include "plugin_helpers.h"
#include "runtime/models/mamba/pipeline.h"
#include "runtime/models/mamba/recurrent_state.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <limits>

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

int32_t positive_numel(const std::vector<int64_t>& shape, int32_t fallback) {
    if (shape.empty())
        return fallback;
    int64_t numel = 1;
    for (const auto dim : shape) {
        if (dim <= 0)
            return fallback;
        if (numel > std::numeric_limits<int32_t>::max() / dim)
            return fallback;
        numel *= dim;
    }
    return static_cast<int32_t>(numel);
}

} // namespace

class MambaPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        if (tp_config.enabled)
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);

        if (tp_config.enabled) {
            opts.distributed_communicator = tp_group.communicator;
            opts.distributed_owner = tp_group.owner;
        }

        const std::string engine_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), "engine_plan", opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.module->stream();

        int32_t d_inner = extract_json_int(ctx.config_json, "intermediate_size", 0);
        if (d_inner == 0)
            d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t state_size = extract_json_int(ctx.config_json, "state_size", 16);
        int32_t conv_kernel = extract_json_int(ctx.config_json, "conv_kernel", 4);
        const int32_t conv_numel =
            positive_numel(loaded.module->tensor_shape("conv_state_0"), d_inner * conv_kernel);
        const int32_t ssm_numel =
            positive_numel(loaded.module->tensor_shape("ssm_state_0"), state_size * d_inner);

        std::vector<MambaRecurrentState::TensorSpec> specs = {
            {"conv_state", {conv_numel}, "present_conv"},
            {"ssm_state", {ssm_numel}, "present_ssm"}};

        auto state = std::make_unique<MambaRecurrentState>(ctx.config.num_layers, specs, stream);
        auto rgc = make_recurrent_gen_config(ctx.config);
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(state), rgc,
                                                   stream, "MambaPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_mamba_plugin, MambaPlugin, "mamba_ssm_recurrent");

} // namespace trtmc
