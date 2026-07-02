/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// VLPlugin: handles "locateanything_vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.

#include "plugin_helpers.h"
#include "runtime/models/locateanything/cuda_stream.h"
#include "runtime/models/locateanything/image_preprocessor.h"
#include "runtime/models/locateanything/pipeline.h"
#include "runtime/models/locateanything/tensor_names.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct TextModuleRuntime {
    ModuleCreateOptions options;
    DistributedRuntimeGroup tp_group;
    std::string engine_section{"engine_plan"};
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

int32_t dim_at(const std::vector<int64_t>& shape, std::size_t idx) {
    return shape.size() > idx ? static_cast<int32_t>(shape[idx]) : 0;
}

int32_t decoder_cache_row_width(const TrtModule& module, const std::string& tensor_name,
                                const BaseConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape(tensor_name), 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

TextModuleRuntime initialize_text_module_runtime(const TensorParallelRuntimeConfig& tp_config) {
    TextModuleRuntime runtime;
    if (!tp_config.enabled)
        return runtime;

    runtime.tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
    runtime.engine_section = tp_engine_section_name(runtime.tp_group.rank);
    return runtime;
}

void configure_text_module_options(TextModuleRuntime& runtime,
                                   const ModuleCreateOptions& base_options,
                                   const TensorParallelRuntimeConfig& tp_config) {
    runtime.options = base_options;
    if (!tp_config.enabled)
        return;

    runtime.options.distributed_communicator = runtime.tp_group.communicator;
    runtime.options.distributed_owner = runtime.tp_group.owner;
}

LocateanythingKvCacheNames build_kv_cache_names(const BaseConfig& config) {
    const auto& io = config.io_map;
    LocateanythingKvCacheNames kv_names;
    for (int32_t i = 0; i < config.num_layers; ++i) {
        kv_names.cache_k.push_back(locateanything_expand_layer_name(io.cache_k_pattern, i));
        kv_names.cache_v.push_back(locateanything_expand_layer_name(io.cache_v_pattern, i));
        kv_names.present_k.push_back(locateanything_expand_layer_name(io.present_k_pattern, i));
        kv_names.present_v.push_back(locateanything_expand_layer_name(io.present_v_pattern, i));
    }
    return kv_names;
}

LoadedModule load_text_module(const PipelineContext& ctx, TextModuleRuntime& runtime,
                              const std::shared_ptr<LocateAnythingCudaStream>& stream) {
    auto loaded =
        load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                                  runtime.engine_section.c_str(), runtime.options);
    loaded.module->keep_alive(stream);
    if (runtime.tp_group.owner)
        loaded.module->keep_alive(runtime.tp_group.owner);
    return loaded;
}

std::unique_ptr<TrtModule>
load_vision_module(IBackend* backend, const BundleFile& bundle, const ModuleCreateOptions& options,
                   const std::shared_ptr<LocateAnythingCudaStream>& stream,
                   bool declared_in_config) {
    auto loaded = try_load_trt_module_from_plan(backend, find_section(bundle, "vision_engine_plan"),
                                                "vision_engine_plan", options);
    if (loaded.module && loaded.module->ok()) {
        loaded.module->keep_alive(stream);
        std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
        return std::move(loaded.module);
    }
    if (declared_in_config) {
        std::cerr << "[trtmc] WARNING: Bundle declares vision engine but "
                     "deserialization failed"
                  << std::endl;
    }
    return nullptr;
}

std::string bundle_section_text(const BundleFile& bundle, const std::string& section_name) {
    const auto* section = find_section(bundle, section_name);
    if (section && !section->empty())
        return std::string(section->begin(), section->end());
    return {};
}

} // namespace

class VLPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        auto text_runtime = initialize_text_module_runtime(tp_config);

        auto shared_stream = std::make_shared<LocateAnythingCudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("VLPlugin: failed to create CUDA stream");

        ModuleCreateOptions opts;
        opts.stream = shared_stream->get();
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        configure_text_module_options(text_runtime, opts, tp_config);

        LocateanythingKvCacheNames kv_names = build_kv_cache_names(ctx.config);

        auto loaded = load_text_module(ctx, text_runtime, shared_stream);

        cudaStream_t stream = loaded.module->stream();
        const std::string cache_k_name =
            kv_names.cache_k.empty() ? std::string("cache_k_0") : kv_names.cache_k.front();
        int32_t kv_dim = decoder_cache_row_width(*loaded.module, cache_k_name, ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<LocateanythingInferenceState> state =
            std::make_unique<LocateanythingKvCache>(ctx.config.num_layers,
                                                    ctx.config.max_cache_length, kv_dim, stream,
                                                    cache_dtype, std::move(kv_names));

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        LocateAnythingConfig vlc;
        vlc.vocab_size = ctx.config.vocab_size;
        vlc.id_bos = ctx.config.id_bos;
        vlc.id_eos = ctx.config.id_eos;
        vlc.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vlc.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vlc.has_position_input = loaded.module->has_input("position_id");

        bool has_vision_engine = extract_json_int(ctx.config_json, "has_vision_engine", 0) != 0;

        // Try to load the vision encoder engine from the bundle.
        std::unique_ptr<TrtModule> vision_module =
            load_vision_module(ctx.backend, ctx.bundle, opts, shared_stream, has_vision_engine);

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        const std::string config_text = bundle_section_text(ctx.bundle, "config.json");
        const std::string preproc_text =
            bundle_section_text(ctx.bundle, "preprocessor_config.json");
        auto vl_preprocess = locateanything_parse_preprocess_config(config_text, preproc_text);

        return std::make_unique<LocateAnythingPipeline>(
            std::move(loaded.module), std::move(vision_module), std::move(state), vlc,
            vl_preprocess, stream, std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_locateanything_plugin, VLPlugin,
                                       "locateanything_vision_language");

} // namespace trtmc
