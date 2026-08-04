/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"
#include "runtime/models/deepseek_ocr/cuda_stream.h"
#include "runtime/models/deepseek_ocr/image_preprocessor.h"
#include "runtime/models/deepseek_ocr/pipeline.h"
#include "runtime/models/deepseek_ocr/tensor_names.h"
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
    TensorParallelRuntimeConfig config;
    config.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    config.enabled = mode == "tensor_parallel" && config.tp_size > 1;
    return config;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

std::string prefill_engine_section_name(const TensorParallelRuntimeConfig& config, int32_t rank) {
    if (!config.enabled)
        return "prefill_engine_plan";
    return "prefill_engine_tp_rank" + std::to_string(rank) + "_plan";
}

int32_t dim_at(const std::vector<int64_t>& shape, std::size_t index) {
    return shape.size() > index ? static_cast<int32_t>(shape[index]) : 0;
}

int32_t decoder_cache_row_width(const TrtModule& module, const std::string& tensor_name,
                                const BaseConfig& config) {
    const auto shape = module.tensor_shape(tensor_name);
    const int32_t from_engine =
        shape.size() == 4 ? dim_at(shape, 1) * dim_at(shape, 3) : dim_at(shape, 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

TextModuleRuntime initialize_text_module_runtime(const TensorParallelRuntimeConfig& config) {
    TextModuleRuntime runtime;
    if (!config.enabled)
        return runtime;
    runtime.tp_group = initialize_tensor_parallel_group(config.tp_size);
    runtime.engine_section = tp_engine_section_name(runtime.tp_group.rank);
    return runtime;
}

void configure_text_module_options(TextModuleRuntime& runtime,
                                   const ModuleCreateOptions& base_options,
                                   const TensorParallelRuntimeConfig& config) {
    runtime.options = base_options;
    if (!config.enabled)
        return;
    runtime.options.distributed_communicator = runtime.tp_group.communicator;
    runtime.options.distributed_owner = runtime.tp_group.owner;
}

DeepseekOcrKvCacheNames build_kv_cache_names(const BaseConfig& config) {
    const auto& io = config.io_map;
    DeepseekOcrKvCacheNames names;
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        names.cache_k.push_back(deepseek_ocr_expand_layer_name(io.cache_k_pattern, layer));
        names.cache_v.push_back(deepseek_ocr_expand_layer_name(io.cache_v_pattern, layer));
        names.present_k.push_back(deepseek_ocr_expand_layer_name(io.present_k_pattern, layer));
        names.present_v.push_back(deepseek_ocr_expand_layer_name(io.present_v_pattern, layer));
    }
    return names;
}

DualProfileModules load_text_modules(const PipelineContext& ctx, TextModuleRuntime& runtime,
                                     const TensorParallelRuntimeConfig& tp_config,
                                     const std::shared_ptr<DeepseekOcrCudaStream>& stream) {
    auto decode_loaded =
        load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, runtime.engine_section),
                                  runtime.engine_section.c_str(), runtime.options);
    if (decode_loaded.prefill)
        throw std::runtime_error("DeepSeek-OCR native decode engine must be single-profile");

    const auto prefill_section = prefill_engine_section_name(tp_config, runtime.tp_group.rank);
    auto prefill_loaded =
        load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, prefill_section),
                                  prefill_section.c_str(), runtime.options);
    if (prefill_loaded.prefill)
        throw std::runtime_error("DeepSeek-OCR native prefill engine must be single-profile");

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

std::unique_ptr<TrtModule> load_vision_module(IBackend* backend, const BundleFile& bundle,
                                              const ModuleCreateOptions& options,
                                              const std::shared_ptr<DeepseekOcrCudaStream>& stream,
                                              bool declared_in_config) {
    auto loaded = try_load_trt_module_from_plan(backend, find_section(bundle, "vision_engine_plan"),
                                                "vision_engine_plan", options);
    if (loaded.module && loaded.module->ok()) {
        loaded.module->keep_alive(stream);
        std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
        return std::move(loaded.module);
    }
    if (declared_in_config) {
        throw std::runtime_error("DeepSeek-OCR bundle declares an unreadable vision engine");
    }
    return nullptr;
}

bool engine_uses_native_kv(const TrtModule& module, const DeepseekOcrKvCacheNames& names) {
    const bool has_write = module.has_input(names.cache_write_indices);
    const bool has_length = module.has_input(names.key_value_lengths);
    if (has_write != has_length)
        throw std::runtime_error("DeepSeek-OCR native engine must expose both KV scalar inputs");
    return has_write;
}

void validate_native_engine(const PipelineContext& ctx, const TrtModule& module,
                            const DeepseekOcrKvCacheNames& names,
                            const TensorParallelRuntimeConfig& tp_config, DType cache_dtype) {
    if (!engine_uses_native_kv(module, names))
        throw std::runtime_error("DeepSeek-OCR legacy KV engines are no longer supported");
    if (cache_dtype != DType::kBFloat16)
        throw std::runtime_error("DeepSeek-OCR native KV runtime requires BF16");
    if (names.cache_k.empty())
        throw std::runtime_error("DeepSeek-OCR native engine has no cache inputs");

    const int32_t local_kv_heads =
        tp_config.enabled ? ctx.config.num_kv_heads / tp_config.tp_size : ctx.config.num_kv_heads;
    const int32_t head_dim = ctx.config.head_dim > 0
                                 ? ctx.config.head_dim
                                 : ctx.config.hidden_size / ctx.config.num_heads;
    const std::vector<int64_t> expected{1, local_kv_heads, ctx.config.max_cache_length, head_dim};
    if (module.tensor_shape(names.cache_k.front()) != expected)
        throw std::runtime_error("DeepSeek-OCR native KV shape does not match rank-local "
                                 "full-context capacity");
}

void validate_native_bundle(const PipelineContext& ctx, const TrtModule& decode,
                            const TrtModule& prefill, const DeepseekOcrKvCacheNames& names,
                            const TensorParallelRuntimeConfig& tp_config, DType cache_dtype) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("DeepSeek-OCR bundle is missing native KV metadata");
    }
    validate_native_engine(ctx, decode, names, tp_config, cache_dtype);
    validate_native_engine(ctx, prefill, names, tp_config, cache_dtype);
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument("DeepSeek-OCR native KV always allocates the model's complete "
                                    "context; kv_cache_size_bytes is not supported");
    }
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("DeepSeek-OCR native KV byte accounting overflow");
    return lhs * rhs;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream stream;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    stream.setf(std::ios::fixed);
    stream.precision(2);
    stream << static_cast<double>(bytes) / kGiB << " GiB";
    return stream.str();
}

void admit_native_kv_allocation(const PipelineContext& ctx, int32_t local_kv_dim,
                                DType cache_dtype) {
    const auto required = checked_multiply(
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
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("DeepSeek-OCR CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }

    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (required > available) {
        throw std::runtime_error("DeepSeek-OCR native KV admission failed before allocation: "
                                 "required=" +
                                 format_bytes(required) + ", free=" + format_bytes(free) +
                                 ", reserve=" + format_bytes(reserve));
    }
}

int32_t prefill_profile_capacity(const TrtModule& module) {
    const auto shape =
        module.input_profile_shape("token_id", module.profile_idx(), ProfileShapeSelector::kMax);
    const auto capacity = dim_at(shape, 0);
    if (capacity <= 0)
        throw std::runtime_error("DeepSeek-OCR native prefill profile has invalid capacity");
    return capacity;
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
        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        auto text_runtime = initialize_text_module_runtime(tp_config);
        auto shared_stream = std::make_shared<DeepseekOcrCudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("DeepSeek-OCR failed to create CUDA stream");

        ModuleCreateOptions options;
        options.stream = shared_stream->get();
        options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        options.cuda_graphs = ctx.cuda_graphs;
        configure_text_module_options(text_runtime, options, tp_config);

        auto kv_names = build_kv_cache_names(ctx.config);
        auto loaded = load_text_modules(ctx, text_runtime, tp_config, shared_stream);
        const bool has_vision_engine =
            extract_json_bool(ctx.config_json, "has_vision_engine", false);
        auto vision_options = options;
        vision_options.distributed_communicator = nullptr;
        vision_options.distributed_owner.reset();
        auto vision_module = load_vision_module(ctx.backend, ctx.bundle, vision_options,
                                                shared_stream, has_vision_engine);

        const auto cache_k_name =
            kv_names.cache_k.empty() ? std::string("cache_k_0") : kv_names.cache_k.front();
        const int32_t kv_dim = decoder_cache_row_width(*loaded.decode, cache_k_name, ctx.config);
        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        validate_native_bundle(ctx, *loaded.decode, *loaded.prefill, kv_names, tp_config,
                               cache_dtype);
        admit_native_kv_allocation(ctx, kv_dim, cache_dtype);

        const auto stream = loaded.decode->stream();
        auto state =
            std::make_unique<DeepseekOcrKvCache>(ctx.config.num_layers, ctx.config.max_cache_length,
                                                 kv_dim, stream, cache_dtype, std::move(kv_names));
        if (!state->ok())
            throw std::runtime_error("DeepSeek-OCR native KV allocation failed");

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        DeepseekOcrConfig vl_config;
        vl_config.vocab_size = ctx.config.vocab_size;
        vl_config.id_bos = ctx.config.id_bos;
        vl_config.id_eos = ctx.config.id_eos;
        vl_config.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vl_config.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vl_config.has_position_input = loaded.decode->has_input("position_id");
        vl_config.num_layers = ctx.config.num_layers;
        vl_config.prefill_max_length = prefill_profile_capacity(*loaded.prefill);
        vl_config.present_k_pattern = ctx.config.io_map.present_k_pattern;
        vl_config.present_v_pattern = ctx.config.io_map.present_v_pattern;

        const auto config_text = bundle_section_text(ctx.bundle, "config.json");
        const auto preprocessor_text = bundle_section_text(ctx.bundle, "preprocessor_config.json");
        auto preprocess = deepseek_ocr_parse_preprocess_config(config_text, preprocessor_text);

        return std::make_unique<DeepseekOcrPipeline>(
            std::move(loaded.decode), std::move(vision_module), std::move(state), vl_config,
            preprocess, stream, std::move(tokenizer), ctx.bundle.info.model_id, nullptr,
            std::move(loaded.prefill));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_deepseek_ocr_plugin, VLPlugin,
                                       "deepseek_ocr_vision_language");

} // namespace trtmc
