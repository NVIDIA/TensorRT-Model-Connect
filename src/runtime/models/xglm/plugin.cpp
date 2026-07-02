/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// DecoderPlugin: handles this model-owned decoder runtime strategy.
// Standard attention-based decoder with device-resident KV cache.

#include "plugin_helpers.h"
#include "runtime/models/xglm/chat_templates.h"
#include "runtime/models/xglm/pipeline.h"
#include "runtime/models/xglm/tensor_names.h"
#include "runtime/models/xglm/triattention_kv_cache.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
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
    const int32_t static_rows = dim_at(module.tensor_shape(cache_name), 0);
    if (static_rows > 0)
        return static_rows;

    if (profile_idx >= 0 && profile_idx < module.optimization_profile_count()) {
        const int32_t max_rows = dim_at(
            module.input_profile_shape(cache_name, profile_idx, ProfileShapeSelector::kMax), 0);
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
                                const XglmKvCacheNames& kv_names, DType cache_dtype,
                                const XglmTriAttentionConfig& tri_cfg, int32_t kv_dim) {
    KvCacheRuntimeSizing sizing;
    const auto elem_bytes = static_cast<std::uint64_t>(dtype_size(cache_dtype));
    sizing.row_bytes = static_cast<std::uint64_t>(ctx.config.num_layers) *
                       static_cast<std::uint64_t>(kv_dim) * elem_bytes * 2ULL;
    if (sizing.row_bytes == 0)
        throw std::runtime_error("Computed zero bytes per KV row");

    const int32_t bundle_max_rows = ctx.config.max_cache_length;
    sizing.runtime_rows = bundle_max_rows;
    sizing.cache_bytes = static_cast<std::uint64_t>(bundle_max_rows) * sizing.row_bytes;

    if (ctx.kv_cache_size_bytes == 0)
        return sizing;

    if (!cache_input_supports_runtime_rows(module, kv_names.cache_k.front())) {
        throw std::runtime_error(
            "This bundle was not built with runtime-resizable KV cache support. "
            "Rebuild with trtmc build --dynamic-kv-cache to use --kv-cache-size.");
    }

    const std::uint64_t requested_rows_u64 = ctx.kv_cache_size_bytes / sizing.row_bytes;
    if (requested_rows_u64 == 0) {
        throw std::runtime_error("--kv-cache-size is smaller than one KV row (" +
                                 format_bytes(sizing.row_bytes) + ")");
    }

    std::uint64_t runtime_rows_u64 = requested_rows_u64;
    if (runtime_rows_u64 > static_cast<std::uint64_t>(bundle_max_rows)) {
        runtime_rows_u64 = static_cast<std::uint64_t>(bundle_max_rows);
        sizing.clamped_to_bundle_max = true;
    }
    if (runtime_rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Resolved KV cache rows exceed int32 runtime limits");
    }

    sizing.runtime_rows = static_cast<int32_t>(runtime_rows_u64);
    sizing.cache_bytes = runtime_rows_u64 * sizing.row_bytes;
    sizing.override_applied = true;

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

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);
        apply_text_trace_from_registry(ctx.runtime_config);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const auto& io = ctx.config.io_map;
        XglmKvCacheNames kv_names;
        build_kv_names(ctx, io, kv_names);

        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        XglmTriAttentionConfig tri_cfg = xglm_parse_triattention_bundle_config(
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

        const int32_t kv_dim = cache_row_dim_from_module(metadata_module, kv_names.cache_k.front());
        const auto sizing = resolve_kv_cache_runtime_sizing(ctx, metadata_module, kv_names,
                                                            cache_dtype, tri_cfg, kv_dim);

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

        auto state =
            build_inference_state(ctx, sizing, tri_cfg, cache_dtype, kv_dim, kv_names, stream);
        log_kv_cache_sizing(ctx, sizing, state.get());

        XglmTextGenConfig tgc;
        populate_text_gen_config(ctx, tgc, io, decoders.front(), ctx.runtime_config);
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

        return std::make_unique<XglmTextGenerationPipeline>(
            std::move(decoders), std::move(state), tgc, stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(prefill_module), nullptr,
            tp_runtime.group.owner);
    }

  private:
    static std::unique_ptr<TrtModule>
    load_split_prefill_module(const PipelineContext& ctx, cudaStream_t stream, const IoMap& io,
                              const XglmKvCacheNames& kv_names, int32_t& prefill_profile_idx,
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
                               XglmKvCacheNames& kv_names) {
        kv_names.position_id = io.position_id;
        kv_names.attention_mask = io.attention_mask;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(xglm_expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(xglm_expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(xglm_expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(xglm_expand_layer_name(io.present_v_pattern, i));
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

    static std::vector<XglmTextGenerationPipeline::DecoderContext>
    build_decoder_contexts(BackendProfileModules profile_modules, int32_t runtime_rows,
                           const DecoderProfileRoles& profile_roles,
                           std::unique_ptr<TrtModule>& prefill_module) {
        std::vector<XglmTextGenerationPipeline::DecoderContext> decoders;
        decoders.reserve(profile_modules.modules.size());
        for (const auto& profile : profile_roles.decode_profiles) {
            if (profile.kv_rows > runtime_rows && !decoders.empty())
                break;
            auto* found = find_profile_module(profile_modules, profile.profile_idx);
            if (found == nullptr || !found->module)
                continue;
            found->module->set_timing_label("engine_plan:decode");
            decoders.push_back(XglmTextGenerationPipeline::DecoderContext{
                profile.kv_rows, std::move(found->module)});
        }

        extract_engine_plan_prefill_module(profile_modules, profile_roles, prefill_module);

        if (decoders.empty())
            throw std::runtime_error("No decoder profile available for engine_plan");
        return decoders;
    }

    static std::unique_ptr<XglmInferenceState>
    build_inference_state(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                          XglmTriAttentionConfig& tri_cfg, DType cache_dtype, int32_t kv_dim,
                          XglmKvCacheNames& kv_names, cudaStream_t stream) {
        std::unique_ptr<XglmInferenceState> state;
        if (tri_cfg.enabled) {
            auto* stats_sec = find_section(ctx.bundle, tri_cfg.stats_section);
            if (stats_sec == nullptr || stats_sec->empty())
                throw std::runtime_error("TriAttention stats section is missing: " +
                                         tri_cfg.stats_section);
            std::string stats_json(stats_sec->begin(), stats_sec->end());
            XglmTriAttentionStats tri_stats = xglm_parse_triattention_stats_json(
                stats_json, ctx.config.num_heads, ctx.config.num_kv_heads, ctx.config.num_layers);
            state = std::make_unique<XglmTriAttentionKvCache>(
                ctx.config.num_layers, ctx.config.num_kv_heads, sizing.runtime_rows, kv_dim, stream,
                std::move(tri_cfg), std::move(tri_stats), cache_dtype, std::move(kv_names));
        } else {
            state = std::make_unique<XglmKvCache>(ctx.config.num_layers, sizing.runtime_rows,
                                                  kv_dim, stream, cache_dtype, std::move(kv_names));
        }
        if (!state->ok())
            throw std::runtime_error("Failed to create XglmKvCache");
        return state;
    }

    static void log_kv_cache_sizing(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                                    XglmInferenceState* state) {
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
    populate_text_gen_config(const PipelineContext& ctx, XglmTextGenConfig& tgc, const IoMap& io,
                             const XglmTextGenerationPipeline::DecoderContext& first_dec,
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

    static void apply_chat_template_format(const BundleFile& bundle, XglmTextGenConfig& tgc) {
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
        tgc.chat_template_format = xglm_detect_chat_template_format(chat_tpl);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_xglm_plugin, DecoderPlugin,
                                       "xglm_decoder_kv_cache");

} // namespace trtmc
