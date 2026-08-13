/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// DecoderPlugin: handles this model-owned decoder runtime strategy.
// Standard attention-based decoder with device-resident KV cache.

#include "plugin_helpers.h"
#include "runtime/models/stablelm/chat_templates.h"
#include "runtime/models/stablelm/pipeline.h"
#include "runtime/models/stablelm/tensor_names.h"
#include "runtime/models/stablelm/triattention_kv_cache.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

struct KvCacheRuntimeSizing {
    int32_t runtime_rows{0};
    std::uint64_t row_bytes{0};
    std::uint64_t cache_bytes{0};
    bool override_applied{false};
    bool clamped_to_bundle_max{false};
};

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct TensorParallelRuntime {
    TensorParallelRuntimeConfig config;
    DistributedRuntimeGroup group;
};

struct DecoderProfileInfo {
    int32_t profile_idx{0};
    int32_t kv_rows{0};
};

struct DecoderProfileRoles {
    int32_t prefill_profile_idx{-1};
    int32_t prefill_max_length{0};
    std::vector<DecoderProfileInfo> decode_profiles;
};

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const int64_t value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t cache_row_dim_from_module(const TrtModule& module, const std::string& tensor_name) {
    const auto row_dim = [](const std::vector<int64_t>& shape) -> int32_t {
        if (shape.size() == 2)
            return dim_at(shape, 1);
        if (shape.size() == 4) {
            const int32_t heads = dim_at(shape, 1);
            const int32_t head_dim = dim_at(shape, 3);
            if (heads > 0 && head_dim > 0 &&
                heads <= std::numeric_limits<int32_t>::max() / head_dim) {
                return heads * head_dim;
            }
        }
        return -1;
    };

    const int32_t static_dim = row_dim(module.tensor_shape(tensor_name));
    if (static_dim > 0)
        return static_dim;
    const int32_t profile_count = module.optimization_profile_count();
    for (int32_t profile_idx = 0; profile_idx < profile_count; ++profile_idx) {
        const int32_t profile_dim = row_dim(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax));
        if (profile_dim > 0)
            return profile_dim;
    }
    throw std::runtime_error("Unable to infer KV row width from engine tensor '" + tensor_name +
                             "'");
}

bool cache_input_is_dynamic(const TrtModule& module, const std::string& tensor_name) {
    const auto shape = module.tensor_shape(tensor_name);
    return !shape.empty() && shape[0] == -1;
}

bool cache_input_supports_runtime_rows(const TrtModule& module, const std::string& tensor_name) {
    if (!cache_input_is_dynamic(module, tensor_name))
        return false;
    const int32_t num_profiles = module.optimization_profile_count();
    if (num_profiles <= 0)
        return false;
    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const int32_t min_rows = dim_at(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMin), 0);
        const int32_t max_rows = dim_at(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax), 0);
        if (min_rows > 0 && max_rows > min_rows)
            return true;
    }
    return false;
}

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

int32_t profile_token_max_length(const TrtModule& module, const std::string& token_id_name,
                                 int32_t profile_idx) {
    return dim_at(
        module.input_profile_shape(token_id_name, profile_idx, ProfileShapeSelector::kMax), 0);
}

int32_t profile_cache_rows(const TrtModule& module, const std::string& cache_name,
                           int32_t profile_idx, int32_t fallback_rows) {
    const auto cache_rows = [](const std::vector<int64_t>& shape) {
        return dim_at(shape, shape.size() == 4 ? 2 : 0);
    };
    const int32_t static_rows = cache_rows(module.tensor_shape(cache_name));
    if (static_rows > 0)
        return static_rows;

    if (profile_idx >= 0 && profile_idx < module.optimization_profile_count()) {
        const int32_t max_rows = cache_rows(
            module.input_profile_shape(cache_name, profile_idx, ProfileShapeSelector::kMax));
        if (max_rows > 0)
            return max_rows;
    }
    return fallback_rows;
}

DecoderProfileRoles detect_decoder_profile_roles(const TrtModule& module,
                                                 const std::string& token_id_name,
                                                 const std::string& cache_k_name,
                                                 int32_t fallback_rows) {
    DecoderProfileRoles roles;
    const int32_t num_profiles = module.optimization_profile_count();
    if (num_profiles <= 0) {
        roles.decode_profiles.push_back(DecoderProfileInfo{0, fallback_rows});
        return roles;
    }

    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const int32_t token_max = profile_token_max_length(module, token_id_name, profile_idx);
        if (token_max > 1) {
            if (token_max > roles.prefill_max_length) {
                roles.prefill_profile_idx = profile_idx;
                roles.prefill_max_length = token_max;
            }
            continue;
        }

        roles.decode_profiles.push_back(DecoderProfileInfo{
            profile_idx, profile_cache_rows(module, cache_k_name, profile_idx, fallback_rows)});
    }

    if (roles.decode_profiles.empty()) {
        const int32_t fallback_profile =
            roles.prefill_profile_idx >= 0 ? roles.prefill_profile_idx : 0;
        roles.decode_profiles.push_back(DecoderProfileInfo{
            fallback_profile,
            profile_cache_rows(module, cache_k_name, fallback_profile, fallback_rows)});
    }

    return roles;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream oss;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        oss << (static_cast<double>(bytes) / kGiB) << " GiB";
        return oss.str();
    }
    if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        oss << (static_cast<double>(bytes) / kMiB) << " MiB";
        return oss.str();
    }
    oss.unsetf(std::ios::floatfield);
    oss.precision(6);
    oss << bytes << " B";
    return oss.str();
}

bool engine_uses_native_kv_updates(const TrtModule& module, const StablelmKvCacheNames& kv_names) {
    const bool has_write_indices = module.has_input(kv_names.cache_write_indices);
    const bool has_kv_lengths = module.has_input(kv_names.key_value_lengths);
    if (has_write_indices != has_kv_lengths) {
        throw std::runtime_error(
            "Stablelm native KV engine must expose both cache_write_indices and "
            "key_value_lengths");
    }
    return has_write_indices;
}

void validate_native_kv_marker(const std::string& config_json, bool engine_uses_native_kv) {
    const bool declares_native_kv = extract_json_bool(config_json, "native_kv_cache", false);
    const bool has_version =
        config_json.find("\"native_kv_contract_version\"") != std::string::npos;
    if (!declares_native_kv && !has_version && !engine_uses_native_kv)
        return;
    if (!declares_native_kv || !engine_uses_native_kv ||
        extract_json_int(config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("Stablelm native KV metadata does not match the engine contract");
    }
}

bool validate_native_kv_runtime(const PipelineContext& ctx, const TrtModule& module,
                                const StablelmKvCacheNames& kv_names, DType cache_dtype,
                                const StablelmTriAttentionConfig& tri_cfg,
                                const TensorParallelRuntimeConfig& tp_cfg) {
    const bool native_kv = engine_uses_native_kv_updates(module, kv_names);
    validate_native_kv_marker(ctx.config_json, native_kv);
    if (!native_kv)
        return false;

    if (cache_dtype != DType::kFloat16 || tri_cfg.enabled || tp_cfg.enabled) {
        throw std::runtime_error(
            "Stablelm native KV requires FP16, single-GPU, non-TriAttention runtime");
    }
    if (ctx.config.head_dim <= 0) {
        throw std::runtime_error("Stablelm native KV requires a positive head_dim");
    }
    const std::vector<int64_t> expected_shape{1, ctx.config.num_kv_heads,
                                              ctx.config.max_cache_length, ctx.config.head_dim};
    if (module.tensor_shape(kv_names.cache_k.front()) != expected_shape) {
        throw std::runtime_error("Stablelm native KV cache shape does not match bundle geometry");
    }
    return true;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("Stablelm native KV byte accounting overflow");
    return lhs * rhs;
}

void admit_native_kv_allocation(const PipelineContext& ctx, bool native_kv,
                                const KvCacheRuntimeSizing& sizing) {
    if (!native_kv)
        return;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Stablelm native KV CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }

    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (sizing.cache_bytes > available) {
        throw std::runtime_error(
            "Stablelm native KV cache admission failed before allocation: capacity=" +
            std::to_string(ctx.config.max_cache_length) +
            " tokens, required=" + format_bytes(sizing.cache_bytes) +
            ", free=" + format_bytes(free) + ", reserve=" + format_bytes(reserve));
    }
}

void reject_native_kv_size_override(const PipelineContext& ctx) {
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "Stablelm native TensorRT KV cache allocates the model's complete fixed "
            "capacity; kv_cache_size_bytes is not supported");
    }
}

void apply_runtime_kv_size_override(const PipelineContext& ctx, const TrtModule& module,
                                    const StablelmKvCacheNames& kv_names,
                                    const StablelmTriAttentionConfig& tri_cfg,
                                    int32_t bundle_max_rows, KvCacheRuntimeSizing& sizing) {
    if (!cache_input_supports_runtime_rows(module, kv_names.cache_k.front())) {
        throw std::runtime_error(
            "This bundle was not built with runtime-resizable KV cache support. "
            "Rebuild with trtmc build --dynamic-kv-cache to use --kv-cache-size.");
    }

    const std::uint64_t requested_rows = ctx.kv_cache_size_bytes / sizing.row_bytes;
    if (requested_rows == 0) {
        throw std::runtime_error("--kv-cache-size is smaller than one KV row (" +
                                 format_bytes(sizing.row_bytes) + ")");
    }

    std::uint64_t runtime_rows = requested_rows;
    if (runtime_rows > static_cast<std::uint64_t>(bundle_max_rows)) {
        runtime_rows = static_cast<std::uint64_t>(bundle_max_rows);
        sizing.clamped_to_bundle_max = true;
    }
    if (runtime_rows > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Resolved KV cache rows exceed int32 runtime limits");
    }

    sizing.runtime_rows = static_cast<int32_t>(runtime_rows);
    sizing.cache_bytes = runtime_rows * sizing.row_bytes;
    sizing.override_applied = true;

    if (tri_cfg.enabled && sizing.runtime_rows < tri_cfg.kv_budget) {
        const auto minimum_bytes = static_cast<std::uint64_t>(tri_cfg.kv_budget) * sizing.row_bytes;
        throw std::runtime_error(
            "--kv-cache-size resolves to " + std::to_string(sizing.runtime_rows) +
            " rows, but this TriAttention bundle needs at least " +
            std::to_string(tri_cfg.kv_budget) + " rows (" + format_bytes(minimum_bytes) + ")");
    }
}

KvCacheRuntimeSizing resolve_kv_cache_runtime_sizing(
    const PipelineContext& ctx, const TrtModule& module, const StablelmKvCacheNames& kv_names,
    DType cache_dtype, const StablelmTriAttentionConfig& tri_cfg, int32_t kv_dim, bool native_kv) {
    KvCacheRuntimeSizing sizing;
    const int32_t bundle_max_rows = ctx.config.max_cache_length;
    if (ctx.config.num_layers <= 0 || kv_dim <= 0 || bundle_max_rows <= 0)
        throw std::runtime_error("Stablelm KV geometry must be positive");
    sizing.row_bytes = checked_multiply(
        checked_multiply(checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                          static_cast<std::uint64_t>(kv_dim)),
                         static_cast<std::uint64_t>(dtype_size(cache_dtype))),
        2);
    sizing.runtime_rows = bundle_max_rows;
    sizing.cache_bytes =
        checked_multiply(static_cast<std::uint64_t>(bundle_max_rows), sizing.row_bytes);

    if (native_kv) {
        reject_native_kv_size_override(ctx);
        return sizing;
    }

    if (ctx.kv_cache_size_bytes == 0)
        return sizing;

    apply_runtime_kv_size_override(ctx, module, kv_names, tri_cfg, bundle_max_rows, sizing);
    return sizing;
}

} // namespace

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);
        apply_text_trace_from_registry(ctx.runtime_config);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const auto& io = ctx.config.io_map;
        StablelmKvCacheNames kv_names;
        build_kv_names(ctx, io, kv_names);

        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        StablelmTriAttentionConfig tri_cfg = stablelm_parse_triattention_bundle_config(
            ctx.config_json, ctx.config.max_cache_length, ctx.runtime_config);

        TensorParallelRuntime tp_runtime;
        tp_runtime.config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (tp_runtime.config.enabled)
            tp_runtime.group = initialize_tensor_parallel_group(tp_runtime.config.tp_size);

        const std::string engine_section = tp_runtime.config.enabled
                                               ? tp_engine_section_name(tp_runtime.group.rank)
                                               : std::string("engine_plan");
        auto profile_modules =
            load_decoder_profile_modules(ctx, engine_section, nullptr, &tp_runtime);
        if (profile_modules.modules.empty())
            throw std::runtime_error("No decoder engine profiles were loaded");
        TrtModule& metadata_module = *profile_modules.modules.front().module;

        const bool native_kv = validate_native_kv_runtime(ctx, metadata_module, kv_names,
                                                          cache_dtype, tri_cfg, tp_runtime.config);
        const int32_t kv_dim = cache_row_dim_from_module(metadata_module, kv_names.cache_k.front());
        const auto sizing = resolve_kv_cache_runtime_sizing(
            ctx, metadata_module, kv_names, cache_dtype, tri_cfg, kv_dim, native_kv);

        const auto decode_profile_roles = detect_decoder_profile_roles(
            metadata_module, io.token_id, kv_names.cache_k.front(), ctx.config.max_cache_length);

        std::unique_ptr<TrtModule> prefill_module;
        auto decoders = build_decoder_contexts(std::move(profile_modules), sizing.runtime_rows,
                                               decode_profile_roles, prefill_module);
        cudaStream_t stream = decoders.front().module->stream();

        int32_t prefill_profile_idx = decode_profile_roles.prefill_profile_idx;
        int32_t prefill_max_length = decode_profile_roles.prefill_max_length;
        std::string prefill_log_label;
        if (!tp_runtime.config.enabled) {
            auto split_prefill_module =
                load_split_prefill_module(ctx, stream, io, kv_names, prefill_profile_idx,
                                          prefill_max_length, prefill_log_label);
            if (split_prefill_module)
                prefill_module = std::move(split_prefill_module);
        }

        // Split prefill deserialization can consume additional execution-context
        // memory. Admit the KV allocation against the free memory that remains
        // after every engine/context needed by this pipeline has been loaded.
        admit_native_kv_allocation(ctx, native_kv, sizing);
        auto state =
            build_inference_state(ctx, sizing, tri_cfg, cache_dtype, kv_dim, kv_names, stream);
        log_kv_cache_sizing(ctx, sizing, state.get());

        StablelmTextGenConfig tgc;
        populate_text_gen_config(ctx, tgc, io, decoders.front(), ctx.runtime_config);
        apply_chat_template_format(ctx.bundle, tgc);
        // Wire batched prefill. Native TensorRT KV engines update the shared
        // aliased cache in place; legacy engines copy prefill outputs into it.
        tgc.prefill_max_length = prefill_max_length;
        tgc.prefill_profile_index = prefill_profile_idx;
        tgc.prefill_log_label = std::move(prefill_log_label);
        tgc.num_layers = ctx.config.num_layers;
        tgc.kv_dim = kv_dim;
        tgc.present_k_pattern = io.present_k_pattern;
        tgc.present_v_pattern = io.present_v_pattern;

        return std::make_unique<StablelmTextGenerationPipeline>(
            std::move(decoders), std::move(state), tgc, stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(prefill_module), nullptr,
            tp_runtime.group.owner);
    }

  private:
    static std::unique_ptr<TrtModule>
    load_split_prefill_module(const PipelineContext& ctx, cudaStream_t stream, const IoMap& io,
                              const StablelmKvCacheNames& kv_names, int32_t& prefill_profile_idx,
                              int32_t& prefill_max_length, std::string& prefill_log_label) {
        if (find_section(ctx.bundle, "prefill_engine_plan") == nullptr)
            return nullptr;

        auto split_prefill_modules =
            load_decoder_profile_modules(ctx, "prefill_engine_plan", stream, nullptr);
        if (split_prefill_modules.modules.empty())
            return nullptr;

        const auto prefill_roles =
            detect_decoder_profile_roles(*split_prefill_modules.modules.front().module, io.token_id,
                                         kv_names.cache_k.front(), ctx.config.max_cache_length);
        prefill_profile_idx = prefill_roles.prefill_profile_idx;
        prefill_max_length = prefill_roles.prefill_max_length;
        auto prefill_module = extract_prefill_module(std::move(split_prefill_modules),
                                                     prefill_roles, "prefill_engine_plan");
        if (prefill_module)
            prefill_log_label = "prefill engine";
        return prefill_module;
    }

    static void apply_text_trace_from_registry(const config::ConfigBundle* cfg) {
        if (cfg == nullptr)
            return;
        try {
            apply_text_trace_config_from_registry(
                cfg->get<std::string>("text_trace", "step_trace_path"),
                cfg->get<std::int32_t>("text_trace", "step_trace_start_pos"),
                cfg->get<std::int32_t>("text_trace", "step_trace_end_pos"),
                cfg->get<std::int32_t>("text_trace", "step_trace_topk"));
        } catch (const std::exception&) {
            // Schema not registered or type mismatch — leave disabled.
        }
    }

    static BackendProfileModules
    load_decoder_profile_modules(const PipelineContext& ctx, const std::string& section_name,
                                 cudaStream_t stream, const TensorParallelRuntime* tp_runtime) {
        auto* plan = find_section(ctx.bundle, section_name);
        if (plan == nullptr || plan->empty())
            throw std::runtime_error(section_name + " section is missing");
        if (ctx.backend == nullptr)
            throw std::runtime_error("No backend loaded");

        auto profile_rows = extract_json_int_array(ctx.config_json, "dynamic_kv_profile_rows", 16);
        const int32_t profile_candidates =
            profile_rows.empty() ? 2 : static_cast<int32_t>(profile_rows.size() + 1);
        std::vector<int32_t> profile_indices;
        profile_indices.reserve(static_cast<std::size_t>(profile_candidates));
        for (int32_t i = 0; i < profile_candidates; ++i)
            profile_indices.push_back(i);

        ModuleCreateOptions opts;
        opts.stream = stream;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        if (tp_runtime != nullptr && tp_runtime->config.enabled) {
            opts.distributed_communicator = tp_runtime->group.communicator;
            opts.distributed_owner = tp_runtime->group.owner;
        }

        const auto t0 = std::chrono::steady_clock::now();
        auto modules =
            ctx.backend->create_profile_modules(plan->data(), plan->size(), opts, profile_indices);
        const auto t1 = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        log_trt_load_timing(section_name.c_str(), load_ms, plan->size());
        for (auto& entry : modules.modules) {
            entry.module->set_timing_label(entry.profile_idx == 0 ? section_name + ":profile0"
                                                                  : section_name + ":decode");
        }
        return modules;
    }

    static void build_kv_names(const PipelineContext& ctx, const IoMap& io,
                               StablelmKvCacheNames& kv_names) {
        kv_names.position_id = io.position_id;
        kv_names.attention_mask = io.attention_mask;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(stablelm_expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(stablelm_expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(stablelm_expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(stablelm_expand_layer_name(io.present_v_pattern, i));
        }
    }

    static std::unique_ptr<TrtModule>
    extract_prefill_module(BackendProfileModules profile_modules,
                           const DecoderProfileRoles& profile_roles, const char* section_name) {
        if (profile_roles.prefill_profile_idx < 0)
            return nullptr;
        for (auto& entry : profile_modules.modules) {
            if (entry.profile_idx != profile_roles.prefill_profile_idx)
                continue;
            entry.module->set_timing_label(std::string(section_name) + ":prefill");
            return std::move(entry.module);
        }
        return nullptr;
    }

    static BackendProfileModule* find_profile_module(BackendProfileModules& profile_modules,
                                                     int32_t profile_idx) {
        auto found = std::find_if(
            profile_modules.modules.begin(), profile_modules.modules.end(),
            [&](const BackendProfileModule& entry) { return entry.profile_idx == profile_idx; });
        if (found == profile_modules.modules.end())
            return nullptr;
        return &*found;
    }

    static void extract_engine_plan_prefill_module(BackendProfileModules& profile_modules,
                                                   const DecoderProfileRoles& profile_roles,
                                                   std::unique_ptr<TrtModule>& prefill_module) {
        if (profile_roles.prefill_profile_idx < 0)
            return;
        auto* entry = find_profile_module(profile_modules, profile_roles.prefill_profile_idx);
        if (entry == nullptr || !entry->module)
            return;
        entry->module->set_timing_label("engine_plan:prefill");
        prefill_module = std::move(entry->module);
    }

    static std::vector<StablelmTextGenerationPipeline::DecoderContext>
    build_decoder_contexts(BackendProfileModules profile_modules, int32_t runtime_rows,
                           const DecoderProfileRoles& profile_roles,
                           std::unique_ptr<TrtModule>& prefill_module) {
        std::vector<StablelmTextGenerationPipeline::DecoderContext> decoders;
        decoders.reserve(profile_modules.modules.size());
        for (const auto& profile : profile_roles.decode_profiles) {
            if (profile.kv_rows > runtime_rows && !decoders.empty())
                break;
            auto* found = find_profile_module(profile_modules, profile.profile_idx);
            if (found == nullptr || !found->module)
                continue;
            found->module->set_timing_label("engine_plan:decode");
            decoders.push_back(StablelmTextGenerationPipeline::DecoderContext{
                profile.kv_rows, std::move(found->module)});
        }

        extract_engine_plan_prefill_module(profile_modules, profile_roles, prefill_module);

        if (decoders.empty())
            throw std::runtime_error("No decoder profile available for engine_plan");
        return decoders;
    }

    static std::unique_ptr<StablelmInferenceState>
    build_inference_state(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                          StablelmTriAttentionConfig& tri_cfg, DType cache_dtype, int32_t kv_dim,
                          StablelmKvCacheNames& kv_names, cudaStream_t stream) {
        std::unique_ptr<StablelmInferenceState> state;
        if (tri_cfg.enabled) {
            auto* stats_sec = find_section(ctx.bundle, tri_cfg.stats_section);
            if (stats_sec == nullptr || stats_sec->empty())
                throw std::runtime_error("TriAttention stats section is missing: " +
                                         tri_cfg.stats_section);
            std::string stats_json(stats_sec->begin(), stats_sec->end());
            StablelmTriAttentionStats tri_stats = stablelm_parse_triattention_stats_json(
                stats_json, ctx.config.num_heads, ctx.config.num_kv_heads, ctx.config.num_layers);
            state = std::make_unique<StablelmTriAttentionKvCache>(
                ctx.config.num_layers, ctx.config.num_kv_heads, sizing.runtime_rows, kv_dim, stream,
                std::move(tri_cfg), std::move(tri_stats), cache_dtype, std::move(kv_names));
        } else {
            state =
                std::make_unique<StablelmKvCache>(ctx.config.num_layers, sizing.runtime_rows,
                                                  kv_dim, stream, cache_dtype, std::move(kv_names));
        }
        if (!state->ok())
            throw std::runtime_error("Failed to create StablelmKvCache");
        return state;
    }

    static void log_kv_cache_sizing(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                                    StablelmInferenceState* state) {
        std::cerr << "[trtmc] KV cache rows=" << sizing.runtime_rows
                  << " (bundle max=" << ctx.config.max_cache_length
                  << ", row=" << format_bytes(sizing.row_bytes)
                  << ", cache=" << format_bytes(sizing.cache_bytes) << ", state="
                  << format_bytes(static_cast<std::uint64_t>(state->device_memory_bytes())) << ")";
        if (sizing.override_applied) {
            std::cerr << " [requested=" << format_bytes(ctx.kv_cache_size_bytes) << "]";
            if (sizing.clamped_to_bundle_max)
                std::cerr << " [clamped-to-bundle-max]";
        }
        std::cerr << '\n';
    }

    static void
    populate_text_gen_config(const PipelineContext& ctx, StablelmTextGenConfig& tgc,
                             const IoMap& io,
                             const StablelmTextGenerationPipeline::DecoderContext& first_dec,
                             const config::ConfigBundle* runtime_config) {
        tgc.vocab_size = ctx.config.vocab_size;
        tgc.id_bos = ctx.config.id_bos;
        tgc.id_eos = ctx.config.id_eos;
        tgc.has_position_input = first_dec.module->has_input(io.position_id);
        tgc.token_id_name = io.token_id;
        tgc.logits_output_name = io.logits;
        if (runtime_config == nullptr)
            return;
        try {
            tgc.disable_cuda_graph = runtime_config->get<bool>("runtime", "disable_cuda_graph");
            tgc.prefer_gpu_greedy = runtime_config->get<bool>("runtime", "prefer_gpu_greedy");
            tgc.log_runtime_stats = runtime_config->get<bool>("platform", "trt_log_stderr");
        } catch (const std::exception&) {
            // Schema not registered — stay at defaults.
        }
    }

    static void apply_chat_template_format(const BundleFile& bundle, StablelmTextGenConfig& tgc) {
        std::string chat_tpl;
        auto* tok_cfg_sec = find_section(bundle, "tokenizer_config.json");
        if (tok_cfg_sec != nullptr && !tok_cfg_sec->empty()) {
            const std::string tok_cfg_text(tok_cfg_sec->begin(), tok_cfg_sec->end());
            chat_tpl = extract_json_string(tok_cfg_text, "chat_template", "");
        }
        if (chat_tpl.empty()) {
            auto* tpl_sec = find_section(bundle, "chat_template.jinja");
            if (tpl_sec != nullptr && !tpl_sec->empty())
                chat_tpl.assign(tpl_sec->begin(), tpl_sec->end());
        }
        tgc.chat_template_format = stablelm_detect_chat_template_format(chat_tpl);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_stablelm_plugin, DecoderPlugin,
                                       "stablelm_decoder_kv_cache");

} // namespace trtmc
