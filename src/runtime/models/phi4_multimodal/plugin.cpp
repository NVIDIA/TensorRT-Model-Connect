/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// VLPlugin: handles "phi4_multimodal_vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.

#include "plugin_helpers.h"
#include "runtime/models/phi4_multimodal/cuda_stream.h"
#include "runtime/models/phi4_multimodal/image_preprocessor.h"
#include "runtime/models/phi4_multimodal/pipeline.h"
#include "runtime/models/phi4_multimodal/tensor_names.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

int32_t dim_at(const std::vector<int64_t>& shape, std::size_t idx) {
    return shape.size() > idx ? static_cast<int32_t>(shape[idx]) : 0;
}

int32_t decoder_cache_row_width(const TrtModule& module, const std::string& tensor_name,
                                const BaseConfig& config) {
    const auto shape = module.tensor_shape(tensor_name);
    if (shape.size() == 4 && shape[1] > 0 && shape[3] > 0)
        return static_cast<int32_t>(shape[1] * shape[3]);
    const int32_t from_engine = dim_at(shape, 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

Phi4MultimodalKvCacheNames build_kv_cache_names(const BaseConfig& config) {
    const auto& io = config.io_map;
    Phi4MultimodalKvCacheNames kv_names;
    for (int32_t i = 0; i < config.num_layers; ++i) {
        kv_names.cache_k.push_back(phi4_multimodal_expand_layer_name(io.cache_k_pattern, i));
        kv_names.cache_v.push_back(phi4_multimodal_expand_layer_name(io.cache_v_pattern, i));
        kv_names.present_k.push_back(phi4_multimodal_expand_layer_name(io.present_k_pattern, i));
        kv_names.present_v.push_back(phi4_multimodal_expand_layer_name(io.present_v_pattern, i));
    }
    return kv_names;
}

DualProfileModules load_text_modules(const PipelineContext& ctx, const ModuleCreateOptions& options,
                                     const std::shared_ptr<Phi4MultimodalCudaStream>& stream) {
    auto loaded = load_dual_profile_modules(ctx.backend, find_section(ctx.bundle, "engine_plan"),
                                            "engine_plan", options);
    loaded.decode->keep_alive(stream);
    if (loaded.prefill)
        loaded.prefill->keep_alive(stream);
    if (find_section(ctx.bundle, "prefill_engine_plan") != nullptr) {
        auto split =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "prefill_engine_plan"),
                                      "prefill_engine_plan", options);
        split.module->keep_alive(stream);
        loaded.prefill = std::move(split.module);
    }
    return loaded;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("Phi-4 Multimodal native KV byte accounting overflow");
    return lhs * rhs;
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr std::uint64_t kGiB = 1ULL << 30;
    constexpr std::uint64_t kMiB = 1ULL << 20;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2);
    if (bytes >= kGiB)
        stream << static_cast<double>(bytes) / kGiB << " GiB";
    else if (bytes >= kMiB)
        stream << static_cast<double>(bytes) / kMiB << " MiB";
    else
        stream << bytes << " B";
    return stream.str();
}

void validate_native_module(const TrtModule& module, const Phi4MultimodalKvCacheNames& names,
                            const BaseConfig& config, DType cache_dtype) {
    if (!module.has_input(names.cache_write_indices) ||
        !module.has_input(names.key_value_lengths) || module.has_input("attention_mask")) {
        throw std::runtime_error("Phi-4 Multimodal decoder does not expose the native KV contract");
    }
    if (cache_dtype != DType::kFloat16)
        throw std::runtime_error("Phi-4 Multimodal native KV requires FP16");
    if (names.cache_k.empty())
        throw std::runtime_error("Phi-4 Multimodal native KV has no cache tensors");
    const auto shape = module.tensor_shape(names.cache_k.front());
    if (shape.size() != 4 || shape[0] != 1 ||
        shape[2] != static_cast<int64_t>(config.max_cache_length) || shape[1] <= 0 ||
        shape[3] <= 0 || shape[1] * shape[3] != compute_kv_dim(config)) {
        throw std::runtime_error("Phi-4 Multimodal native KV cache must be [1,Hkv,capacity,D]");
    }
}

void validate_native_bundle(const PipelineContext& ctx, const DualProfileModules& modules,
                            const Phi4MultimodalKvCacheNames& names, DType cache_dtype) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("Phi-4 Multimodal bundle is missing native KV contract metadata");
    }
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "Phi-4 Multimodal native KV allocates the model's complete context; "
            "--kv-cache-size is not supported");
    }
    if (modules.prefill == nullptr)
        throw std::runtime_error(
            "Phi-4 Multimodal native KV bundle requires split prefill/decode engines");
    validate_native_module(*modules.decode, names, ctx.config, cache_dtype);
    validate_native_module(*modules.prefill, names, ctx.config, cache_dtype);
}

std::uint64_t native_cache_bytes(const BaseConfig& config, int32_t kv_dim, DType dtype) {
    auto bytes = checked_multiply(static_cast<std::uint64_t>(config.max_cache_length),
                                  static_cast<std::uint64_t>(config.num_layers));
    bytes = checked_multiply(bytes, static_cast<std::uint64_t>(kv_dim));
    bytes = checked_multiply(bytes, static_cast<std::uint64_t>(dtype_size(dtype)));
    return checked_multiply(bytes, 2);
}

void admit_native_cache_allocation(const BaseConfig& config, std::uint64_t required) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Phi-4 Multimodal CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }
    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto reserve = std::max(kTwoGiB, static_cast<std::uint64_t>(total_bytes) / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (required > available) {
        throw std::runtime_error(
            "Phi-4 Multimodal native KV admission failed before allocation: capacity=" +
            std::to_string(config.max_cache_length) +
            " tokens, required=" + format_bytes(required) + ", free=" + format_bytes(free) +
            ", reserve=" + format_bytes(reserve));
    }
}

std::unique_ptr<TrtModule>
load_vision_module(IBackend* backend, const BundleFile& bundle, const ModuleCreateOptions& options,
                   const std::shared_ptr<Phi4MultimodalCudaStream>& stream,
                   bool declared_in_config) {
    const auto* plan = find_section(bundle, "vision_engine_plan");
    const bool required = declared_in_config || plan != nullptr;
    if (!required)
        return nullptr;

    auto loaded = load_trt_module_from_plan(backend, plan, "vision_engine_plan", options);
    loaded.module->keep_alive(stream);
    std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
    return std::move(loaded.module);
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
        if (extract_json_string(ctx.config_json, "tensor_parallel_mode", "single") != "single" ||
            extract_json_int(ctx.config_json, "tensor_parallel_size", 1) != 1) {
            throw std::runtime_error(
                "Phi-4 Multimodal native KV does not support tensor parallel runtime");
        }

        auto shared_stream = std::make_shared<Phi4MultimodalCudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("VLPlugin: failed to create CUDA stream");

        ModuleCreateOptions opts;
        opts.stream = shared_stream->get();
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        Phi4MultimodalKvCacheNames kv_names = build_kv_cache_names(ctx.config);

        auto loaded = load_text_modules(ctx, opts, shared_stream);

        cudaStream_t stream = loaded.decode->stream();
        const std::string cache_k_name =
            kv_names.cache_k.empty() ? std::string("cache_k_0") : kv_names.cache_k.front();
        int32_t kv_dim = decoder_cache_row_width(*loaded.decode, cache_k_name, ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        validate_native_bundle(ctx, loaded, kv_names, cache_dtype);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        Phi4MultimodalConfig vlc;
        vlc.vocab_size = ctx.config.vocab_size;
        vlc.id_bos = ctx.config.id_bos;
        vlc.id_eos = ctx.config.id_eos;
        vlc.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vlc.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vlc.has_position_input = loaded.decode->has_input("position_id");
        vlc.num_layers = ctx.config.num_layers;
        const auto prefill_max_shape = loaded.prefill->input_profile_shape(
            "token_id", loaded.prefill->profile_idx(), ProfileShapeSelector::kMax);
        vlc.prefill_max_length = dim_at(prefill_max_shape, 0);
        if (vlc.prefill_max_length <= 0)
            throw std::runtime_error(
                "Phi-4 Multimodal native prefill profile has no valid token capacity");
        vlc.present_k_pattern = ctx.config.io_map.present_k_pattern;
        vlc.present_v_pattern = ctx.config.io_map.present_v_pattern;

        const bool has_vision_engine =
            extract_json_bool(ctx.config_json, "has_vision_engine", false);

        // A declared vision engine is part of this pipeline's memory footprint.
        std::unique_ptr<TrtModule> vision_module =
            load_vision_module(ctx.backend, ctx.bundle, opts, shared_stream, has_vision_engine);

        // All engine contexts are resident now. Preserve a safety reserve and
        // fail before any partial full-context cache allocation can occur.
        const auto cache_bytes = native_cache_bytes(ctx.config, kv_dim, cache_dtype);
        admit_native_cache_allocation(ctx.config, cache_bytes);
        std::unique_ptr<Phi4MultimodalInferenceState> state =
            std::make_unique<Phi4MultimodalKvCache>(ctx.config.num_layers,
                                                    ctx.config.max_cache_length, kv_dim, stream,
                                                    cache_dtype, std::move(kv_names));
        if (!state->ok())
            throw std::runtime_error("Phi-4 Multimodal native KV full-context allocation failed");
        std::cerr << "[trtmc] Phi-4 Multimodal native KV capacity=" << ctx.config.max_cache_length
                  << " tokens, cache=" << format_bytes(cache_bytes) << '\n';

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        const std::string config_text = bundle_section_text(ctx.bundle, "config.json");
        const std::string preproc_text =
            bundle_section_text(ctx.bundle, "preprocessor_config.json");
        auto vl_preprocess = phi4_multimodal_parse_preprocess_config(config_text, preproc_text);

        return std::make_unique<Phi4MultimodalPipeline>(
            std::move(loaded.decode), std::move(vision_module), std::move(state), vlc,
            vl_preprocess, stream, std::move(tokenizer), ctx.bundle.info.model_id, nullptr,
            std::move(loaded.prefill));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_phi4_multimodal_plugin, VLPlugin,
                                       "phi4_multimodal_vision_language");

} // namespace trtmc
