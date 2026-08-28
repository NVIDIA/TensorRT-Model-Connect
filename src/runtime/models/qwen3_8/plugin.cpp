/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Qwen38Plugin: handles the Qwen3.8-owned hybrid recurrent strategy.
// Qwen3.8 style models with interleaved attention and Mamba layers,
// using Qwen38KvCache for attention layers and Qwen38RecurrentState for SSM layers.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_8/hybrid_state.h"
#include "runtime/models/qwen3_8/pipeline.h"
#include "runtime/models/qwen3_8/recurrent_state.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct TensorParallelRuntime {
    TensorParallelRuntimeConfig config;
    DistributedRuntimeGroup group;
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

class Qwen38Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        TensorParallelRuntime tp_runtime;
        tp_runtime.config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (tp_runtime.config.enabled)
            tp_runtime.group = initialize_tensor_parallel_group(tp_runtime.config.tp_size);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        if (tp_runtime.config.enabled) {
            opts.distributed_communicator = tp_runtime.group.communicator;
            opts.distributed_owner = tp_runtime.group.owner;
        }

        const std::string engine_section = tp_runtime.config.enabled
                                               ? tp_engine_section_name(tp_runtime.group.rank)
                                               : std::string("engine_plan");
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), engine_section.c_str(), opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.module->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);

        int32_t num_attention_layers = extract_json_int(ctx.config_json, "num_attention_layers", 0);
        int32_t num_mamba_layers = extract_json_int(ctx.config_json, "num_mamba_layers", 0);
        int32_t d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t mamba_d_state = extract_json_int(ctx.config_json, "mamba_d_state", 128);
        int32_t mamba_d_conv = extract_json_int(ctx.config_json, "mamba_d_conv", 4);
        int32_t mamba_nheads = extract_json_int(ctx.config_json, "mamba_nheads", 0);
        int32_t mamba_head_dim = extract_json_int(ctx.config_json, "mamba_head_dim", 0);
        int32_t conv_dim = extract_json_int(ctx.config_json, "conv_dim", d_inner);

        // Every value above defaults to 0 when the bundle omits its key, and a
        // zero silently produces a degenerate state rather than an error: no
        // attention layers means an empty KV cache whose ok() is trivially
        // true, and zero mamba heads means zero-element SSM tensors bound to an
        // engine that expects real state. Reject them at load time instead.
        const auto require_positive = [](int32_t value, const char* key) {
            if (value <= 0)
                throw std::runtime_error(std::string("Qwen3.8 bundle config is missing or has a "
                                                     "non-positive value for '") +
                                         key + "'");
        };
        require_positive(num_attention_layers, "num_attention_layers");
        require_positive(num_mamba_layers, "num_mamba_layers");
        require_positive(mamba_nheads, "mamba_nheads");
        require_positive(mamba_head_dim, "mamba_head_dim");
        require_positive(mamba_d_state, "mamba_d_state");
        require_positive(mamba_d_conv, "mamba_d_conv");
        require_positive(kv_dim, "num_key_value_heads * head_dim");

        // Qwen38KvCache for the attention layers
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        auto cache = std::make_unique<Qwen38KvCache>(
            num_attention_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!cache->ok())
            throw std::runtime_error("Failed to create Qwen38KvCache for hybrid model");

        // Qwen38RecurrentState for the Mamba/SSM layers (conv_state + ssm_state)
        int32_t effective_conv_dim = (conv_dim > 0) ? conv_dim : d_inner;
        int64_t conv_elems = static_cast<int64_t>(effective_conv_dim) * mamba_d_conv;
        int64_t ssm_elems =
            static_cast<int64_t>(mamba_nheads) * std::max(mamba_head_dim, 1) * mamba_d_state;

        auto ssm =
            std::make_unique<Qwen38RecurrentState>(num_mamba_layers,
                                                   std::vector<Qwen38RecurrentState::TensorSpec>{
                                                       {"conv_state", {conv_elems}, "present_conv"},
                                                       {"ssm_state", {ssm_elems}, "present_ssm"}},
                                                   stream);

        if (!ssm->ok())
            throw std::runtime_error("Failed to create Qwen38RecurrentState for hybrid model");

        auto hybrid = std::make_unique<Qwen38HybridState>(std::move(cache), std::move(ssm));
        if (!hybrid->ok())
            throw std::runtime_error("Failed to create Qwen38HybridState for hybrid model");
        auto rgc = make_recurrent_gen_config(ctx.config);
        rgc.has_position_input = loaded.module->has_input("position_id");
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(hybrid), rgc,
                                                   stream, "HybridPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_8_plugin, Qwen38Plugin,
                                       "qwen3_8_hybrid_mamba_attention");

} // namespace trtmc
