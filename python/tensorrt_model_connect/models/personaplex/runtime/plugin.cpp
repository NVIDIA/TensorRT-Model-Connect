/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// PersonaPlexPlugin: handles "personaplex_speech_to_speech" strategy.
// Speech pipeline with temporal engine, depth engines, and mimi encoder/decoder.

#include "audio_helpers.h"
#include "plugin_helpers.h"
#include "pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const TrtModule& module, int32_t fallback) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : fallback;
}

} // namespace

class PersonaPlexPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto speech_cfg =
            build_speech_config_from_bundle(ctx.bundle, ctx.config_json, ctx.config, ctx.hf_python);
        infer_speech_vocab_sizes(speech_cfg, ctx.config_json, ctx.config);

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        DistributedRuntimeGroup tp_group;
        ModuleCreateOptions temporal_opts = opts;
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            temporal_opts.distributed_communicator = tp_group.communicator;
            temporal_opts.distributed_owner = tp_group.owner;
        }

        const std::string temporal_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto temporal_loaded =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, temporal_section),
                                      "speech temporal", temporal_opts);

        cudaStream_t stream = temporal_loaded.module->stream();
        ModuleCreateOptions chained_opts = opts;
        chained_opts.stream = stream;

        const int32_t temporal_kv_fallback =
            compute_kv_dim_kv_heads(ctx.config, ctx.config.hidden_size);
        int32_t temporal_kv_dim =
            decoder_cache_row_width(*temporal_loaded.module, temporal_kv_fallback);

        DType temporal_cache_dtype = temporal_loaded.module->tensor_dtype("cache_k_0");
        std::unique_ptr<PersonaplexInferenceState> temporal_state =
            std::make_unique<PersonaplexKvCache>(ctx.config.num_layers, ctx.config.max_cache_length,
                                                 temporal_kv_dim, stream, temporal_cache_dtype);
        if (!temporal_state->ok())
            throw std::runtime_error(
                "SpeechPipeline: failed to create temporal PersonaplexKvCache");

        auto depth_engines = load_depth_engines(ctx.backend, ctx.bundle, chained_opts);

        const auto depth_cfg = make_depth_engine_config(ctx.config_json, ctx.config);
        int32_t depth_kv_dim = compute_kv_dim_kv_heads(depth_cfg, depth_cfg.hidden_size);
        DType depth_cache_dtype = depth_engines.empty()
                                      ? cache_dtype_from_precision(ctx.config.precision)
                                      : depth_engines.front()->tensor_dtype("cache_k_0");

        std::unique_ptr<PersonaplexInferenceState> depth_state =
            std::make_unique<PersonaplexKvCache>(depth_cfg.num_layers, depth_cfg.max_cache_length,
                                                 depth_kv_dim, stream, depth_cache_dtype);
        if (!depth_state->ok())
            throw std::runtime_error("SpeechPipeline: failed to create depth PersonaplexKvCache");

        auto mimi_encoder =
            extract_optional_module(ctx.backend, find_section(ctx.bundle, "mimi_encoder_plan"),
                                    "speech mimi_encoder", chained_opts);
        auto mimi_decoder =
            extract_optional_module(ctx.backend, find_section(ctx.bundle, "mimi_decoder_plan"),
                                    "speech mimi_decoder", chained_opts);

        if (ctx.cuda_graphs) {
            temporal_loaded.module->enable_cuda_graph();
            for (auto& depth_engine : depth_engines)
                depth_engine->enable_cuda_graph();
            if (mimi_encoder)
                mimi_encoder->enable_cuda_graph();
            if (mimi_decoder)
                mimi_decoder->enable_cuda_graph();
        }

        return std::make_unique<SpeechPipeline>(
            std::move(mimi_encoder), std::move(temporal_loaded.module), std::move(temporal_state),
            std::move(depth_engines), std::move(depth_state), std::move(mimi_decoder),
            std::move(speech_cfg), stream,
            nullptr, // subprocess_runner: default
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_personaplex_plugin, PersonaPlexPlugin,
                                       "personaplex_speech_to_speech");

} // namespace trtmc
