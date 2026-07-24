/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// QwenDecoderPlugin: Qwen-owned decoder runtime with device-resident KV cache.

#include "plugin_helpers.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/domains/text/dynamic_memory/kv_cache_budget.h"
#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "runtime/models/qwen/chat_templates.h"
#include "runtime/models/qwen/pipeline.h"
#include "runtime/models/qwen/tensor_names.h"
#include "runtime/models/qwen/triattention_kv_cache.h"
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
    int32_t logical_max_sequence{0};
    std::uint64_t row_bytes{0};
    std::uint64_t cache_bytes{0};
    std::uint64_t free_bytes_after_engines{0};
    std::uint64_t total_device_bytes{0};
    std::uint64_t budget_bytes{0};
    std::uint64_t memory_capacity_rows{0};
    std::uint64_t safety_reserve_bytes{0};
    KvCacheBudgetPolicy policy{KvCacheBudgetPolicy::kAuto};
    double fraction{0.0};
    bool dynamic_supported{false};
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

const RuntimeMemoryQualifiedTuple& qualified_runtime_memory_tuple() {
    static const RuntimeMemoryQualifiedTuple tuple = [] {
        RuntimeMemoryQualifiedTuple value;
        value.model_id = "Qwen/Qwen3-0.6B";
        value.revision = "c1899de289a04d12100db370d81485cdf75e47ca";
        value.config_sha256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd";
        value.target = "gb300-trt-11.2";
        value.gpu_architecture = "sm103";
        value.trt_runtime_version = "11.2.0.113";
        value.cuda_runtime_version = "13.3";
        value.cudnn_backend_version = "9.20.0";
        value.cudnn_frontend_revision = "7b9b711c22b6823e87150213ecd8449260db8610";
        value.nvrtc_version = "13.3";
        value.driver_version = "580.105.08";
        value.model_context_limit = 40960;
        value.prefill_chunk_limit = 1024;
        value.active_kv_profile_limits = {128, 256, 512, 1024, 2048, 8192, 32768, 40960};
        return value;
    }();
    return tuple;
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
    const auto width_from_shape = [](const std::vector<int64_t>& shape) {
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
    const int32_t static_dim = width_from_shape(module.tensor_shape(tensor_name));
    if (static_dim > 0)
        return static_dim;
    const int32_t profile_count = module.optimization_profile_count();
    for (int32_t profile_idx = 0; profile_idx < profile_count; ++profile_idx) {
        const int32_t profile_dim = width_from_shape(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax));
        if (profile_dim > 0)
            return profile_dim;
    }
    throw std::runtime_error("Unable to infer KV row width from engine tensor '" + tensor_name +
                             "'");
}

bool cache_input_is_dynamic(const TrtModule& module, const std::string& tensor_name) {
    return module.input_is_dynamic(tensor_name);
}

bool cache_input_supports_runtime_rows(const TrtModule& module, const std::string& tensor_name) {
    if (!cache_input_is_dynamic(module, tensor_name))
        return false;
    const int32_t num_profiles = module.optimization_profile_count();
    if (num_profiles <= 0)
        return false;
    const int32_t sequence_axis = module.input_rank(tensor_name) == 4 ? 2 : 0;
    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const int32_t min_rows =
            dim_at(module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMin),
                   sequence_axis);
        const int32_t max_rows =
            dim_at(module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax),
                   sequence_axis);
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
    const int32_t sequence_axis = module.input_rank(cache_name) == 4 ? 2 : 0;
    if (!module.input_is_dynamic(cache_name)) {
        const int32_t static_rows = dim_at(module.tensor_shape(cache_name), sequence_axis);
        if (static_rows > 0)
            return static_rows;
    }

    if (profile_idx >= 0 && profile_idx < module.optimization_profile_count()) {
        const int32_t max_rows =
            dim_at(module.input_profile_shape(cache_name, profile_idx, ProfileShapeSelector::kMax),
                   sequence_axis);
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

KvCacheRuntimeSizing
resolve_kv_cache_runtime_sizing(const PipelineContext& ctx, const TrtModule& module,
                                const QwenKvCacheNames& kv_names, DType cache_dtype,
                                const QwenTriAttentionConfig& tri_cfg, int32_t kv_dim) {
    KvCacheRuntimeSizing sizing;
    const auto elem_bytes = static_cast<std::uint64_t>(dtype_size(cache_dtype));
    sizing.row_bytes = static_cast<std::uint64_t>(ctx.config.num_layers) *
                       static_cast<std::uint64_t>(kv_dim) * elem_bytes * 2ULL;
    if (sizing.row_bytes == 0)
        throw std::runtime_error("Computed zero bytes per KV row");

    const int32_t bundle_max_rows = ctx.config.max_cache_length;
    sizing.runtime_rows = bundle_max_rows;
    sizing.logical_max_sequence = bundle_max_rows;
    sizing.cache_bytes = static_cast<std::uint64_t>(bundle_max_rows) * sizing.row_bytes;
    sizing.budget_bytes = sizing.cache_bytes;
    sizing.memory_capacity_rows = static_cast<std::uint64_t>(bundle_max_rows);

    // Preserve the established LoadOptions/PipelineContext behavior. Automatic
    // post-load sizing belongs exclusively to the versioned runtime_memory
    // plugin interface below; legacy bundles use their built maximum unless
    // the legacy byte override was explicitly supplied.
    if (ctx.kv_cache_size_bytes == 0)
        return sizing;

    if (!cache_input_supports_runtime_rows(module, kv_names.cache_k.front())) {
        throw std::runtime_error(
            "This bundle was not built with runtime-resizable KV cache support. "
            "Use the model-only build flow for a runtime-memory-qualified model, "
            "or omit the runtime KV override.");
    }

    std::uint64_t runtime_rows = ctx.kv_cache_size_bytes / sizing.row_bytes;
    if (runtime_rows == 0) {
        throw std::runtime_error("--kv-cache-size is smaller than one KV row (" +
                                 format_bytes(sizing.row_bytes) + ")");
    }
    if (runtime_rows > static_cast<std::uint64_t>(bundle_max_rows)) {
        runtime_rows = static_cast<std::uint64_t>(bundle_max_rows);
        sizing.clamped_to_bundle_max = true;
    }
    if (runtime_rows > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("Resolved KV cache rows exceed int32 runtime limits");

    sizing.dynamic_supported = true;
    sizing.policy = KvCacheBudgetPolicy::kBytes;
    sizing.budget_bytes = ctx.kv_cache_size_bytes;
    sizing.memory_capacity_rows = ctx.kv_cache_size_bytes / sizing.row_bytes;
    sizing.runtime_rows = static_cast<int32_t>(runtime_rows);
    sizing.logical_max_sequence = sizing.runtime_rows;
    sizing.cache_bytes = runtime_rows * sizing.row_bytes;
    if (tri_cfg.enabled && sizing.runtime_rows < tri_cfg.kv_budget) {
        const auto minimum_bytes = static_cast<std::uint64_t>(tri_cfg.kv_budget) * sizing.row_bytes;
        throw std::runtime_error(
            "--kv-cache-size resolves to " + std::to_string(sizing.runtime_rows) +
            " rows, but this TriAttention bundle needs at least " +
            std::to_string(tri_cfg.kv_budget) + " rows (" + format_bytes(minimum_bytes) + ")");
    }

    return sizing;
}

} // namespace

class QwenDecoderPlugin final : public IPipelinePlugin, public IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        if (ctx.bundle.info.runtime_memory.present) {
            throw std::runtime_error("Qwen runtime_memory bundle requires plugin interface V1; "
                                     "the loading core is incompatible with this model plugin");
        }
        return create_impl(ctx, nullptr);
    }

    std::unique_ptr<IPipeline>
    create_runtime_memory(const PipelineContext& ctx,
                          const RuntimeMemoryPluginOptionsV1& options) override {
        if (options.struct_size < sizeof(RuntimeMemoryPluginOptionsV1) ||
            options.api_version != kRuntimeMemoryPluginApiVersionV1) {
            throw std::invalid_argument(
                "Qwen runtime-memory plugin options have an incompatible size or API version");
        }
        if (!ctx.bundle.info.runtime_memory.present) {
            throw std::invalid_argument(
                "Qwen runtime-memory plugin interface requires a runtime_memory bundle");
        }
        if (ctx.bundle.info.runtime_memory.native_kv_plugin_abi != 2) {
            throw std::runtime_error("Qwen runtime-memory bundle requires native_kv_plugin_abi=2 "
                                     "(history-only cache inputs with exact-Sq staging outputs)");
        }
        const auto* runtime_backend = dynamic_cast<IRuntimeMemoryBackendV1*>(ctx.backend);
        if (runtime_backend == nullptr || runtime_backend->runtime_memory_api_version() !=
                                              kRuntimeMemoryBackendApiVersionCurrent) {
            throw std::runtime_error(
                "Qualified runtime-memory bundle requires the standard TensorRT "
                "runtime-memory backend v1");
        }
        const auto& expected = qualified_runtime_memory_tuple();
        validate_runtime_memory_qualified_tuple(ctx.bundle.info.runtime_memory, expected);
        validate_runtime_memory_module_residency_calibration(
            ctx.bundle.info.runtime_memory, ctx.bundle);
        return create_impl(ctx, &options);
    }

  private:
    static std::unique_ptr<IPipeline>
    create_impl(const PipelineContext& ctx, const RuntimeMemoryPluginOptionsV1* runtime_options) {
        load_ffi_kernels_from_bundle(ctx.bundle);
        apply_text_trace_from_registry(ctx.runtime_config);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const auto& io = ctx.config.io_map;
        QwenKvCacheNames kv_names;
        build_kv_names(ctx, io, kv_names);

        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        QwenTriAttentionConfig tri_cfg = qwen_parse_triattention_bundle_config(
            ctx.config_json, ctx.config.max_cache_length, ctx.runtime_config);
        const bool triattention_enabled = tri_cfg.enabled;

        TensorParallelRuntime tp_runtime;
        tp_runtime.config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (tp_runtime.config.enabled)
            tp_runtime.group = initialize_tensor_parallel_group(tp_runtime.config.tp_size);

        const std::string engine_section = tp_runtime.config.enabled
                                               ? tp_engine_section_name(tp_runtime.group.rank)
                                               : std::string("engine_plan");
        RuntimeDeviceMemorySnapshot pre_load_memory_snapshot;
        const bool pre_load_memory_snapshot_available = ctx.bundle.info.runtime_memory.present;
        if (pre_load_memory_snapshot_available) {
            pre_load_memory_snapshot = query_runtime_device_memory_snapshot(
                "before runtime-memory Qwen engine deserialization");
        }
        auto profile_modules =
            load_decoder_profile_modules(ctx, engine_section, nullptr, &tp_runtime, kv_names);
        if (profile_modules.modules.empty())
            throw std::runtime_error("No decoder engine profiles were loaded");
        TrtModule& metadata_module = *profile_modules.modules.front().module;

        const int32_t kv_dim = cache_row_dim_from_module(metadata_module, kv_names.cache_k.front());
        const auto decode_profile_roles = detect_decoder_profile_roles(
            metadata_module, io.token_id, kv_names.cache_k.front(), ctx.config.max_cache_length);

        // Load both halves of a split bundle before observing free memory.
        // KV inputs are deferred by ModuleCreateOptions, so this measurement
        // includes engines, contexts, workspaces, and prefill staging, but not
        // a hidden profile-MAX KV allocation.
        cudaStream_t stream = metadata_module.stream();

        int32_t prefill_profile_idx = decode_profile_roles.prefill_profile_idx;
        int32_t prefill_max_length = decode_profile_roles.prefill_max_length;
        std::string prefill_log_label;
        std::unique_ptr<TrtModule> split_prefill_module;
        if (!tp_runtime.config.enabled) {
            split_prefill_module =
                load_split_prefill_module(ctx, stream, io, kv_names, prefill_profile_idx,
                                          prefill_max_length, prefill_log_label);
        }

        if (runtime_options != nullptr) {
            if (triattention_enabled)
                throw std::runtime_error(
                    "Qualified contiguous runtime memory is incompatible with TriAttention");
            if (tp_runtime.config.enabled)
                throw std::runtime_error(
                    "Qualified contiguous runtime memory does not support tensor parallelism");
            if (!split_prefill_module)
                throw std::runtime_error(
                    "Qualified runtime-memory bundle is missing prefill_engine_plan");

            const auto& contract = ctx.bundle.info.runtime_memory;
            validate_runtime_memory_qualified_tuple(contract, qualified_runtime_memory_tuple());
            if (contract.contract_version != 2 || contract.native_kv_plugin_abi != 2 ||
                contract.kv_layout != "contiguous_runtime_v1" || contract.kv_dtype != "bfloat16" ||
                !contract.runtime_owned) {
                throw std::runtime_error(
                    "Qwen runtime cannot consume this runtime_memory contract");
            }
            if (cache_dtype != DType::kBFloat16)
                throw std::runtime_error(
                    "Qualified runtime-memory Qwen bundle requires BF16 KV tensors");
            if (ctx.config.num_kv_heads <= 0 || ctx.config.head_dim <= 0 ||
                kv_dim != ctx.config.num_kv_heads * ctx.config.head_dim) {
                throw std::runtime_error(
                    "Qualified runtime-memory Qwen engine has inconsistent KV dimensions");
            }

            RuntimeKvSetupRequest setup;
            setup.layout.layer_count = static_cast<std::uint32_t>(ctx.config.num_layers);
            setup.layout.kv_head_count = static_cast<std::uint32_t>(ctx.config.num_kv_heads);
            setup.layout.head_dim = static_cast<std::uint32_t>(ctx.config.head_dim);
            setup.layout.capacity_tokens = static_cast<std::uint64_t>(contract.model_context_limit);
            setup.layout.prefill_chunk_limit =
                static_cast<std::uint32_t>(contract.prefill_chunk_limit);
            setup.layout.dtype = cache_dtype;
            setup.layout.names.token_id = io.token_id;
            setup.layout.names.position_id = kv_names.position_id;
            setup.layout.names.history_length = kv_names.history_length;
            setup.layout.names.cache_k = kv_names.cache_k;
            setup.layout.names.cache_v = kv_names.cache_v;
            setup.layout.names.cache_k_output = kv_names.present_k;
            setup.layout.names.cache_v_output = kv_names.present_v;
            setup.expected_active_kv_profile_limits.assign(
                contract.active_kv_profile_limits.begin(), contract.active_kv_profile_limits.end());
            const auto& calibration = contract.module_residency_calibration;
            for (const auto& reserve : calibration.profile_reserves) {
                setup.module_residency_reserve_bytes_by_profile.push_back(
                    reserve.cumulative_reserve_bytes);
            }
            setup.module_residency_plan_set_sha256 = calibration.plan_set_sha256;
            setup.module_residency_cuda_module_loading_mode =
                calibration.cuda_module_loading_mode;
            setup.module_residency_evidence_sha256 = calibration.evidence_sha256;
            setup.expected_kv_bytes_per_token = contract.kv_bytes_per_token;
            setup.request_context_limit = runtime_options->max_sequence_length;
            setup.stream = stream;
            if (runtime_options->kv_cache_memory_policy == KvCacheMemoryPolicy::kBytes) {
                setup.policy.kind = RuntimeKvPolicyKind::kBytes;
                setup.policy.bytes = runtime_options->kv_cache_memory_bytes;
            } else if (runtime_options->kv_cache_memory_policy == KvCacheMemoryPolicy::kFraction) {
                setup.policy.kind = RuntimeKvPolicyKind::kFraction;
                setup.policy.fraction = runtime_options->kv_cache_memory_fraction;
            } else if (runtime_options->kv_cache_memory_policy == KvCacheMemoryPolicy::kAuto) {
                setup.policy.kind = RuntimeKvPolicyKind::kAuto;
                setup.policy.fraction = 0.90;
            } else {
                throw std::invalid_argument(
                    "Qwen runtime-memory plugin received an unknown KV memory policy");
            }

            const auto* decoder_plan = find_section(ctx.bundle, engine_section);
            const auto* prefill_plan = find_section(ctx.bundle, "prefill_engine_plan");
            setup.serialized_plan_bytes =
                static_cast<std::uint64_t>(decoder_plan ? decoder_plan->size() : 0) +
                static_cast<std::uint64_t>(prefill_plan ? prefill_plan->size() : 0);
            setup.pre_load_memory_snapshot = pre_load_memory_snapshot;
            setup.pre_load_memory_snapshot_available = pre_load_memory_snapshot_available;
            setup.roles.push_back(RuntimeKvExecutionRole{
                split_prefill_module.get(), RuntimeKvExecutionRoleKind::kPrefill,
                static_cast<std::uint64_t>(contract.model_context_limit)});
            for (const auto& profile : decode_profile_roles.decode_profiles) {
                auto* entry = find_profile_module(profile_modules, profile.profile_idx);
                if (entry == nullptr || !entry->module)
                    throw std::runtime_error(
                        "Qualified Qwen bundle is missing a decode profile module");
                setup.roles.push_back(
                    RuntimeKvExecutionRole{entry->module.get(), RuntimeKvExecutionRoleKind::kDecode,
                                           static_cast<std::uint64_t>(profile.kv_rows)});
            }

            auto runtime_state = create_runtime_kv_state(setup);
            const int32_t runtime_rows = static_cast<int32_t>(runtime_state->capacity_tokens());
            const int32_t effective_request_limit =
                static_cast<int32_t>(runtime_state->receipt().effective_request_limit);
            const std::string memory_receipt = runtime_state->receipt_json();
            const RuntimeMemoryReceipt admission_receipt = runtime_state->receipt();

            std::unique_ptr<TrtModule> ignored_prefill;
            auto decoders = build_decoder_contexts(std::move(profile_modules), runtime_rows,
                                                   decode_profile_roles, ignored_prefill);
            auto state =
                std::make_unique<QwenKvCache>(ctx.config.num_layers, runtime_rows, kv_dim, stream,
                                              cache_dtype, kv_names, std::move(runtime_state));
            if (!state->ok())
                throw std::runtime_error("Failed to create runtime-owned QwenKvCache");

            QwenTextGenConfig tgc;
            populate_text_gen_config(ctx, tgc, io, decoders.front(), ctx.runtime_config);
            tgc.max_sequence_length = effective_request_limit;
            tgc.runtime_sequence_admission = RuntimeSequenceAdmissionContext{
                admission_receipt.model_context_limit,
                admission_receipt.runtime_kv_capacity_tokens,
                admission_receipt.request_context_limit,
                admission_receipt.kv_bytes_per_token,
                admission_receipt.kv_budget_bytes,
                admission_receipt.kv_reserved_bytes,
            };
            tgc.prefill_max_length = contract.prefill_chunk_limit;
            tgc.prefill_profile_index = prefill_profile_idx;
            tgc.prefill_log_label = std::move(prefill_log_label);
            tgc.num_layers = ctx.config.num_layers;
            tgc.kv_dim = kv_dim;
            tgc.present_k_pattern.clear();
            tgc.present_v_pattern.clear();
            // Correctness uses a history-only cache extent. T=1 uniquely
            // identifies the H=0 cold sentinel; warm invocations require
            // H>0 and T>=max(H,2), while current Sq rows remain separate
            // staging outputs and H+Sq<=R. CUDA graph capture stays disabled
            // until these profile-specific H/T shapes are qualified for replay.
            tgc.disable_cuda_graph = true;
            apply_chat_template_format(ctx.bundle, tgc);

            std::cerr << "[trtmc.memory] " << memory_receipt << '\n';
            return std::make_unique<QwenTextGenerationPipeline>(
                std::move(decoders), std::move(state), tgc, stream, std::move(tokenizer),
                ctx.bundle.info.model_id, nullptr, std::move(split_prefill_module), nullptr,
                tp_runtime.group.owner);
        }

        const auto sizing = resolve_kv_cache_runtime_sizing(ctx, metadata_module, kv_names,
                                                            cache_dtype, tri_cfg, kv_dim);

        std::unique_ptr<TrtModule> prefill_module;
        auto decoders = build_decoder_contexts(std::move(profile_modules), sizing.runtime_rows,
                                               decode_profile_roles, prefill_module);
        if (split_prefill_module)
            prefill_module = std::move(split_prefill_module);

        auto state =
            build_inference_state(ctx, sizing, tri_cfg, cache_dtype, kv_dim, kv_names, stream);
        log_kv_cache_sizing(ctx, sizing, state.get());

        QwenTextGenConfig tgc;
        populate_text_gen_config(ctx, tgc, io, decoders.front(), ctx.runtime_config);
        tgc.max_sequence_length = sizing.logical_max_sequence;
        tgc.kv_cache_compaction = triattention_enabled;
        apply_chat_template_format(ctx.bundle, tgc);
        // Wire batched prefill: the pipeline forwards the whole prompt
        // through `prefill_module` (TRT optimization profile 0) and copies
        // per-layer K/V into the shared cache via write_prefill_kv.
        tgc.prefill_max_length = prefill_max_length;
        tgc.prefill_profile_index = prefill_profile_idx;
        tgc.prefill_log_label = std::move(prefill_log_label);
        tgc.num_layers = ctx.config.num_layers;
        tgc.kv_dim = kv_dim;
        tgc.present_k_pattern = io.present_k_pattern;
        tgc.present_v_pattern = io.present_v_pattern;

        return std::make_unique<QwenTextGenerationPipeline>(
            std::move(decoders), std::move(state), tgc, stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(prefill_module), nullptr,
            tp_runtime.group.owner);
    }

    static std::unique_ptr<TrtModule>
    load_split_prefill_module(const PipelineContext& ctx, cudaStream_t stream, const IoMap& io,
                              const QwenKvCacheNames& kv_names, int32_t& prefill_profile_idx,
                              int32_t& prefill_max_length, std::string& prefill_log_label) {
        if (find_section(ctx.bundle, "prefill_engine_plan") == nullptr)
            return nullptr;

        auto split_prefill_modules =
            load_decoder_profile_modules(ctx, "prefill_engine_plan", stream, nullptr, kv_names);
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
                                 cudaStream_t stream, const TensorParallelRuntime* tp_runtime,
                                 const QwenKvCacheNames& kv_names) {
        auto* plan = find_section(ctx.bundle, section_name);
        if (plan == nullptr || plan->empty())
            throw std::runtime_error(section_name + " section is missing");
        if (ctx.backend == nullptr)
            throw std::runtime_error("No backend loaded");

        auto profile_rows = extract_json_int_array(ctx.config_json, "dynamic_kv_profile_rows", 16);
        int32_t profile_candidates =
            profile_rows.empty() ? 2 : static_cast<int32_t>(profile_rows.size() + 1);
        if (ctx.bundle.info.runtime_memory.present) {
            profile_candidates =
                section_name == "prefill_engine_plan"
                    ? 1
                    : static_cast<int32_t>(
                          ctx.bundle.info.runtime_memory.active_kv_profile_limits.size());
            if (profile_candidates <= 0)
                throw std::runtime_error(
                    "Qualified runtime-memory bundle has no execution profiles");
        }
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
        BackendProfileModules modules;
        if (ctx.bundle.info.runtime_memory.present) {
            auto* runtime_backend = dynamic_cast<IRuntimeMemoryBackendV1*>(ctx.backend);
            if (runtime_backend == nullptr || runtime_backend->runtime_memory_api_version() !=
                                                  kRuntimeMemoryBackendApiVersionCurrent) {
                throw std::runtime_error(
                    "Qualified runtime-memory bundle requires the standard TensorRT "
                    "runtime-memory backend v1");
            }
            RuntimeMemoryModuleOptionsV1 memory_options;
            memory_options.deferred_tensor_names.reserve(
                kv_names.cache_k.size() + kv_names.cache_v.size() + kv_names.present_k.size() +
                kv_names.present_v.size());
            memory_options.deferred_tensor_names.insert(memory_options.deferred_tensor_names.end(),
                                                        kv_names.cache_k.begin(),
                                                        kv_names.cache_k.end());
            memory_options.deferred_tensor_names.insert(memory_options.deferred_tensor_names.end(),
                                                        kv_names.cache_v.begin(),
                                                        kv_names.cache_v.end());
            memory_options.deferred_tensor_names.insert(memory_options.deferred_tensor_names.end(),
                                                        kv_names.present_k.begin(),
                                                        kv_names.present_k.end());
            memory_options.deferred_tensor_names.insert(memory_options.deferred_tensor_names.end(),
                                                        kv_names.present_v.begin(),
                                                        kv_names.present_v.end());
            modules = runtime_backend->create_profile_modules_runtime_memory(
                plan->data(), plan->size(), opts, profile_indices, memory_options);
        } else {
            modules = ctx.backend->create_profile_modules(plan->data(), plan->size(), opts,
                                                          profile_indices);
        }
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
                               QwenKvCacheNames& kv_names) {
        kv_names.position_id = io.position_id;
        kv_names.attention_mask = io.attention_mask;
        kv_names.history_length = io.history_length;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(qwen_expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(qwen_expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(qwen_expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(qwen_expand_layer_name(io.present_v_pattern, i));
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

    static std::vector<QwenTextGenerationPipeline::DecoderContext>
    build_decoder_contexts(BackendProfileModules profile_modules, int32_t runtime_rows,
                           const DecoderProfileRoles& profile_roles,
                           std::unique_ptr<TrtModule>& prefill_module) {
        std::vector<QwenTextGenerationPipeline::DecoderContext> decoders;
        decoders.reserve(profile_modules.modules.size());
        bool covered_runtime_rows = false;
        for (const auto& profile : profile_roles.decode_profiles) {
            auto* found = find_profile_module(profile_modules, profile.profile_idx);
            if (found == nullptr || !found->module)
                continue;
            found->module->set_timing_label("engine_plan:decode");
            decoders.push_back(QwenTextGenerationPipeline::DecoderContext{
                profile.kv_rows, std::move(found->module)});
            // Keep the first profile whose MAX covers the runtime allocation.
            // A non-bucket size such as 100 rows still needs the 128-row
            // profile even though the external buffer itself contains 100 rows.
            if (profile.kv_rows >= runtime_rows) {
                covered_runtime_rows = true;
                break;
            }
        }

        extract_engine_plan_prefill_module(profile_modules, profile_roles, prefill_module);

        if (decoders.empty())
            throw std::runtime_error("No decoder profile available for engine_plan");
        if (!covered_runtime_rows) {
            throw std::runtime_error(
                "No decoder optimization profile covers runtime KV cache rows=" +
                std::to_string(runtime_rows));
        }
        return decoders;
    }

    static std::unique_ptr<QwenInferenceState>
    build_inference_state(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                          QwenTriAttentionConfig& tri_cfg, DType cache_dtype, int32_t kv_dim,
                          QwenKvCacheNames& kv_names, cudaStream_t stream) {
        std::unique_ptr<QwenInferenceState> state;
        if (tri_cfg.enabled) {
            auto* stats_sec = find_section(ctx.bundle, tri_cfg.stats_section);
            if (stats_sec == nullptr || stats_sec->empty())
                throw std::runtime_error("TriAttention stats section is missing: " +
                                         tri_cfg.stats_section);
            std::string stats_json(stats_sec->begin(), stats_sec->end());
            QwenTriAttentionStats tri_stats = qwen_parse_triattention_stats_json(
                stats_json, ctx.config.num_heads, ctx.config.num_kv_heads, ctx.config.num_layers);
            state = std::make_unique<QwenTriAttentionKvCache>(
                ctx.config.num_layers, ctx.config.num_kv_heads, sizing.runtime_rows, kv_dim, stream,
                std::move(tri_cfg), std::move(tri_stats), cache_dtype, std::move(kv_names));
        } else {
            state = std::make_unique<QwenKvCache>(ctx.config.num_layers, sizing.runtime_rows,
                                                  kv_dim, stream, cache_dtype, std::move(kv_names));
        }
        if (!state->ok())
            throw std::runtime_error("Failed to create QwenKvCache");
        return state;
    }

    static void log_kv_cache_sizing(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                                    QwenInferenceState* state) {
        if (!sizing.dynamic_supported) {
            std::cerr << "[trtmc] KV cache policy=legacy-static"
                      << " engine-cap=" << ctx.config.max_cache_length
                      << " physical-kv-rows=" << sizing.runtime_rows
                      << " runtime-max-sequence=" << sizing.logical_max_sequence
                      << " row=" << format_bytes(sizing.row_bytes)
                      << " allocated=" << format_bytes(sizing.cache_bytes) << " state="
                      << format_bytes(static_cast<std::uint64_t>(state->device_memory_bytes()))
                      << '\n';
            return;
        }

        std::cerr << "[trtmc] KV cache policy=" << kv_cache_budget_policy_name(sizing.policy)
                  << " free-after-engines=" << format_bytes(sizing.free_bytes_after_engines)
                  << " device-total=" << format_bytes(sizing.total_device_bytes)
                  << " safety-reserve=" << format_bytes(sizing.safety_reserve_bytes)
                  << " budget=" << format_bytes(sizing.budget_bytes);
        if (sizing.policy == KvCacheBudgetPolicy::kAuto ||
            sizing.policy == KvCacheBudgetPolicy::kFraction) {
            std::cerr << " fraction=" << (sizing.fraction * 100.0) << "%";
        } else {
            std::cerr << " requested=" << format_bytes(ctx.kv_cache_size_bytes);
        }
        std::cerr << " row=" << format_bytes(sizing.row_bytes)
                  << " memory-capacity=" << sizing.memory_capacity_rows
                  << " engine-cap=" << ctx.config.max_cache_length
                  << " physical-kv-rows=" << sizing.runtime_rows
                  << " runtime-max-sequence=" << sizing.logical_max_sequence
                  << " allocated=" << format_bytes(sizing.cache_bytes) << " state="
                  << format_bytes(static_cast<std::uint64_t>(state->device_memory_bytes()));
        if (sizing.clamped_to_bundle_max)
            std::cerr << " [clamped-to-engine-cap]";
        std::cerr << '\n';
    }

    static void
    populate_text_gen_config(const PipelineContext& ctx, QwenTextGenConfig& tgc, const IoMap& io,
                             const QwenTextGenerationPipeline::DecoderContext& first_dec,
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

    static void apply_chat_template_format(const BundleFile& bundle, QwenTextGenConfig& tgc) {
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
        tgc.chat_template_format = qwen_detect_chat_template_format(chat_tpl);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen_plugin, QwenDecoderPlugin,
                                       "qwen_decoder_kv_cache");

} // namespace trtmc
