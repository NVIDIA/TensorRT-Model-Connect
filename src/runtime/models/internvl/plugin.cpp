/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// VLPlugin: handles "internvl_vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.

#include "plugin_helpers.h"
#include "runtime/models/internvl/cuda_stream.h"
#include "runtime/models/internvl/image_preprocessor.h"
#include "runtime/models/internvl/pipeline.h"
#include "runtime/models/internvl/tensor_names.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
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
    const auto shape = module.tensor_shape(tensor_name);
    const int32_t from_engine =
        shape.size() == 4 ? dim_at(shape, 1) * dim_at(shape, 3) : dim_at(shape, 1);
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

InternvlKvCacheNames build_kv_cache_names(const BaseConfig& config) {
    const auto& io = config.io_map;
    InternvlKvCacheNames kv_names;
    for (int32_t i = 0; i < config.num_layers; ++i) {
        kv_names.cache_k.push_back(internvl_expand_layer_name(io.cache_k_pattern, i));
        kv_names.cache_v.push_back(internvl_expand_layer_name(io.cache_v_pattern, i));
        kv_names.present_k.push_back(internvl_expand_layer_name(io.present_k_pattern, i));
        kv_names.present_v.push_back(internvl_expand_layer_name(io.present_v_pattern, i));
    }
    return kv_names;
}

std::string prefill_engine_section_name(const TensorParallelRuntimeConfig& tp_config,
                                        int32_t rank) {
    if (!tp_config.enabled)
        return "prefill_engine_plan";
    return "prefill_engine_tp_rank" + std::to_string(rank) + "_plan";
}

DualProfileModules load_text_modules(const PipelineContext& ctx, TextModuleRuntime& runtime,
                                     const TensorParallelRuntimeConfig& tp_config,
                                     const std::shared_ptr<InternVlCudaStream>& stream) {
    auto decode_loaded =
        load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                                  runtime.engine_section.c_str(), runtime.options);
    if (decode_loaded.prefill)
        throw std::runtime_error("InternVL native KV decode engine must be single-profile");

    const std::string prefill_section =
        prefill_engine_section_name(tp_config, runtime.tp_group.rank);
    auto prefill_loaded =
        load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, prefill_section),
                                  prefill_section.c_str(), runtime.options);
    if (prefill_loaded.prefill)
        throw std::runtime_error("InternVL native KV prefill engine must be single-profile");

    DualProfileModules loaded;
    loaded.decode = std::move(decode_loaded.decode);
    loaded.prefill = std::move(prefill_loaded.decode);
    loaded.decode->keep_alive(stream);
    loaded.prefill->keep_alive(stream);
    if (runtime.tp_group.owner) {
        loaded.decode->keep_alive(runtime.tp_group.owner);
        loaded.prefill->keep_alive(runtime.tp_group.owner);
    }
    return loaded;
}

bool engine_uses_native_kv_updates(const TrtModule& module, const InternvlKvCacheNames& names) {
    const bool has_write = module.has_input(names.cache_write_indices);
    const bool has_lengths = module.has_input(names.key_value_lengths);
    if (has_write != has_lengths)
        throw std::runtime_error("InternVL native KV engine must expose both scalar inputs");
    return has_write;
}

void validate_native_engine(const PipelineContext& ctx, const TrtModule& module,
                            const InternvlKvCacheNames& names,
                            const TensorParallelRuntimeConfig& tp_config, DType cache_dtype) {
    if (!engine_uses_native_kv_updates(module, names))
        throw std::runtime_error("InternVL legacy KV engines are no longer supported");
    if (cache_dtype != DType::kBFloat16)
        throw std::runtime_error("InternVL native KV runtime requires BF16");
    if (names.cache_k.empty())
        throw std::runtime_error("InternVL native KV engine has no cache inputs");

    const auto shape = module.tensor_shape(names.cache_k.front());
    const int32_t local_kv_heads =
        tp_config.enabled ? ctx.config.num_kv_heads / tp_config.tp_size : ctx.config.num_kv_heads;
    const int32_t head_dim = ctx.config.head_dim > 0
                                 ? ctx.config.head_dim
                                 : ctx.config.hidden_size / ctx.config.num_heads;
    const std::vector<int64_t> expected{1, local_kv_heads, ctx.config.max_cache_length, head_dim};
    if (shape != expected)
        throw std::runtime_error(
            "InternVL native KV cache shape does not match rank-local full capacity");
}

void validate_native_bundle(const PipelineContext& ctx, const TrtModule& decode,
                            const TrtModule& prefill, const InternvlKvCacheNames& names,
                            const TensorParallelRuntimeConfig& tp_config, DType cache_dtype) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1)
        throw std::runtime_error("InternVL bundle is missing native KV contract metadata");
    validate_native_engine(ctx, decode, names, tp_config, cache_dtype);
    validate_native_engine(ctx, prefill, names, tp_config, cache_dtype);
    if (ctx.kv_cache_size_bytes != 0)
        throw std::invalid_argument(
            "InternVL native KV allocates the model's complete fixed capacity; "
            "kv_cache_size_bytes is not supported");
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("InternVL native KV byte accounting overflow");
    return lhs * rhs;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream oss;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << static_cast<double>(bytes) / kGiB << " GiB";
    return oss.str();
}

void admit_native_kv_allocation(const PipelineContext& ctx, int32_t local_kv_dim,
                                DType cache_dtype) {
    const auto bytes = checked_multiply(
        checked_multiply(
            checked_multiply(
                checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                 static_cast<std::uint64_t>(ctx.config.max_cache_length)),
                static_cast<std::uint64_t>(local_kv_dim)),
            static_cast<std::uint64_t>(dtype_size(cache_dtype))),
        2);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("InternVL CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (bytes > available)
        throw std::runtime_error(
            "InternVL native KV admission failed before allocation: required=" +
            format_bytes(bytes) + ", free=" + format_bytes(free) +
            ", reserve=" + format_bytes(reserve));
}

int32_t prefill_profile_capacity(const TrtModule& module) {
    const auto shape =
        module.input_profile_shape("token_id", module.profile_idx(), ProfileShapeSelector::kMax);
    const int32_t value = dim_at(shape, 0);
    if (value <= 0)
        throw std::runtime_error("InternVL native prefill engine has invalid profile capacity");
    return value;
}

std::unique_ptr<TrtModule> load_vision_module(IBackend* backend, const BundleFile& bundle,
                                              const ModuleCreateOptions& options,
                                              const std::shared_ptr<InternVlCudaStream>& stream,
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

        auto shared_stream = std::make_shared<InternVlCudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("VLPlugin: failed to create CUDA stream");

        ModuleCreateOptions opts;
        opts.stream = shared_stream->get();
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        configure_text_module_options(text_runtime, opts, tp_config);

        InternvlKvCacheNames kv_names = build_kv_cache_names(ctx.config);

        auto loaded = load_text_modules(ctx, text_runtime, tp_config, shared_stream);
        const bool has_vision_engine =
            extract_json_int(ctx.config_json, "has_vision_engine", 0) != 0;
        auto vision_options = opts;
        vision_options.distributed_communicator = nullptr;
        vision_options.distributed_owner.reset();
        auto vision_module = load_vision_module(ctx.backend, ctx.bundle, vision_options,
                                                shared_stream, has_vision_engine);

        cudaStream_t stream = loaded.decode->stream();
        const std::string cache_k_name =
            kv_names.cache_k.empty() ? std::string("cache_k_0") : kv_names.cache_k.front();
        int32_t kv_dim = decoder_cache_row_width(*loaded.decode, cache_k_name, ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        validate_native_bundle(ctx, *loaded.decode, *loaded.prefill, kv_names, tp_config,
                               cache_dtype);
        admit_native_kv_allocation(ctx, kv_dim, cache_dtype);
        std::unique_ptr<InternvlInferenceState> state =
            std::make_unique<InternvlKvCache>(ctx.config.num_layers, ctx.config.max_cache_length,
                                              kv_dim, stream, cache_dtype, std::move(kv_names));
        if (!state->ok())
            throw std::runtime_error("Failed to allocate InternVL native KV cache");

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        InternVlConfig vlc;
        vlc.vocab_size = ctx.config.vocab_size;
        vlc.id_bos = ctx.config.id_bos;
        vlc.id_eos = ctx.config.id_eos;
        vlc.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vlc.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vlc.has_position_input = loaded.decode->has_input("position_id");
        vlc.num_layers = ctx.config.num_layers;
        vlc.prefill_max_length = prefill_profile_capacity(*loaded.prefill);
        vlc.present_k_pattern = ctx.config.io_map.present_k_pattern;
        vlc.present_v_pattern = ctx.config.io_map.present_v_pattern;

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        const std::string config_text = bundle_section_text(ctx.bundle, "config.json");
        const std::string preproc_text =
            bundle_section_text(ctx.bundle, "preprocessor_config.json");
        auto vl_preprocess = internvl_parse_preprocess_config(config_text, preproc_text);

        return std::make_unique<InternVlPipeline>(
            std::move(loaded.decode), std::move(vision_module), std::move(state), vlc,
            vl_preprocess, stream, std::move(tokenizer), ctx.bundle.info.model_id, nullptr,
            std::move(loaded.prefill));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_internvl_plugin, VLPlugin,
                                       "internvl_vision_language");

} // namespace trtmc
