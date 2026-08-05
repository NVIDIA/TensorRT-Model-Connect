/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NemotronHPlugin: handles the Nemotron-H-owned hybrid recurrent strategy.
// Nemotron-H style models with interleaved attention and Mamba layers,
// using NemotronHKvCache for attention layers and NemotronHRecurrentState for SSM layers.

#include "bundle/bundle_format.h"
#include "plugin_helpers.h"
#include "runtime/models/nemotron_h/hybrid_state.h"
#include "runtime/models/nemotron_h/pipeline.h"
#include "runtime/models/nemotron_h/recurrent_state.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const int64_t value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t cache_row_dim_from_module(const TrtModule& module, const std::string& tensor_name) {
    const int32_t static_dim = dim_at(module.tensor_shape(tensor_name), 1);
    if (static_dim > 0)
        return static_dim;
    const int32_t profile_count = module.optimization_profile_count();
    for (int32_t profile_idx = 0; profile_idx < profile_count; ++profile_idx) {
        const int32_t profile_dim = dim_at(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax), 1);
        if (profile_dim > 0)
            return profile_dim;
    }
    throw std::runtime_error("Unable to infer KV row width from engine tensor '" + tensor_name +
                             "'");
}

int64_t positive_numel_from_module(const TrtModule& module, const std::string& tensor_name) {
    const auto shape = module.tensor_shape(tensor_name);
    if (shape.empty())
        throw std::runtime_error("Unable to infer positive element count from engine tensor '" +
                                 tensor_name + "'");

    int64_t numel = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || numel > std::numeric_limits<int64_t>::max() / dim)
            throw std::runtime_error(
                "Unable to infer positive element count from engine tensor '" + tensor_name +
                "'");
        numel *= dim;
    }
    return numel;
}

} // namespace

class NemotronHPlugin final : public IPipelinePlugin {
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
        const std::vector<char>* engine_plan = find_section(ctx.bundle, engine_section);
        std::vector<char> lazy_engine_plan;
        if (engine_plan == nullptr) {
            const auto section_info = std::find_if(
                ctx.bundle.info.sections.begin(), ctx.bundle.info.sections.end(),
                [&engine_section](const BundleSectionInfo& section) {
                    return section.name == engine_section;
                });
            if (section_info != ctx.bundle.info.sections.end()) {
                lazy_engine_plan = ReadBundleSection(ctx.bundle_path, *section_info);
                engine_plan = &lazy_engine_plan;
            }
        }
        auto loaded = load_trt_module_from_plan(
            ctx.backend, engine_plan, engine_section.c_str(), opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.module->stream();

        int32_t num_attention_layers = extract_json_int(ctx.config_json, "num_attention_layers", 0);
        int32_t num_mamba_layers = extract_json_int(ctx.config_json, "num_mamba_layers", 0);
        const int32_t kv_dim = num_attention_layers > 0
                                   ? cache_row_dim_from_module(*loaded.module, "cache_k_0")
                                   : compute_kv_dim(ctx.config);
        int32_t d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t mamba_d_state = extract_json_int(ctx.config_json, "mamba_d_state", 128);
        int32_t mamba_d_conv = extract_json_int(ctx.config_json, "mamba_d_conv", 4);
        int32_t mamba_nheads = extract_json_int(ctx.config_json, "mamba_nheads", 0);
        int32_t mamba_head_dim = extract_json_int(ctx.config_json, "mamba_head_dim", 0);
        int32_t conv_dim = extract_json_int(ctx.config_json, "conv_dim", d_inner);

        // NemotronHKvCache for the attention layers
        DType cache_dtype = num_attention_layers > 0
                                ? loaded.module->tensor_dtype("cache_k_0")
                                : cache_dtype_from_precision(ctx.config.precision);
        auto cache = std::make_unique<NemotronHKvCache>(
            num_attention_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!cache->ok())
            throw std::runtime_error("Failed to create NemotronHKvCache for hybrid model");

        // NemotronHRecurrentState for the Mamba/SSM layers (conv_state + ssm_state)
        int32_t effective_conv_dim = (conv_dim > 0) ? conv_dim : d_inner;
        const int64_t conv_elems =
            num_mamba_layers > 0
                ? positive_numel_from_module(*loaded.module, "conv_state_0")
                : static_cast<int64_t>(effective_conv_dim) * mamba_d_conv;
        const int64_t ssm_elems =
            num_mamba_layers > 0
                ? positive_numel_from_module(*loaded.module, "ssm_state_0")
                : static_cast<int64_t>(mamba_nheads) * std::max(mamba_head_dim, 1) * mamba_d_state;

        auto ssm = std::make_unique<NemotronHRecurrentState>(
            num_mamba_layers,
            std::vector<NemotronHRecurrentState::TensorSpec>{
                {"conv_state", {conv_elems}, "present_conv"},
                {"ssm_state", {ssm_elems}, "present_ssm"}},
            stream);

        auto hybrid = std::make_unique<NemotronHHybridState>(std::move(cache), std::move(ssm));
        auto rgc = make_recurrent_gen_config(ctx.config);
        rgc.has_position_input = loaded.module->has_input("position_id");
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(hybrid), rgc,
                                                   stream, "HybridPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_nemotron_h_plugin, NemotronHPlugin,
                                       "nemotron_h_hybrid_mamba_attention");

} // namespace trtmc
