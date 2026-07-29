/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// DecoderPlugin: handles this model-owned decoder runtime strategy.
// Standard attention-based decoder with device-resident KV cache.

#include "plugin_helpers.h"
#include "runtime/models/granite/chat_templates.h"
#include "runtime/models/granite/pipeline.h"
#include "runtime/models/granite/tensor_names.h"
#include "trtmc/config/config_bundle.h"
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
};

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
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

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

int32_t profile_token_length(const TrtModule& module, const std::string& token_id_name,
                             ProfileShapeSelector selector) {
    return dim_at(module.input_profile_shape(token_id_name, 0, selector), 0);
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

void require_native_kv_inputs(const TrtModule& module, const GraniteKvCacheNames& kv_names) {
    const bool has_write_indices = module.has_input(kv_names.cache_write_indices);
    const bool has_kv_lengths = module.has_input(kv_names.key_value_lengths);
    if (!has_write_indices || !has_kv_lengths)
        throw std::runtime_error("Granite bundles must use TensorRT native KV inputs "
                                 "cache_write_indices and key_value_lengths");
}

void validate_native_kv_marker(const std::string& config_json) {
    const bool declares_native_kv = extract_json_bool(config_json, "native_kv_cache", false);
    if (!declares_native_kv ||
        extract_json_int(config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("Granite native KV metadata does not match the engine contract");
    }
}

bool valid_native_cache_shape(const std::vector<int64_t>& shape, const PipelineContext& ctx) {
    if (shape.size() != 4)
        return false;
    if (shape[0] != 1 || shape[1] != ctx.config.num_kv_heads)
        return false;
    if (shape[2] != ctx.config.max_cache_length)
        return false;
    return shape[3] == 64 || shape[3] == 128;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("Granite native KV byte accounting overflow");
    return lhs * rhs;
}

void admit_native_kv_allocation(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Granite native KV CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }

    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (sizing.cache_bytes > available) {
        throw std::runtime_error(
            "Granite native KV cache admission failed before allocation: capacity=" +
            std::to_string(ctx.config.max_cache_length) +
            " tokens, required=" + format_bytes(sizing.cache_bytes) +
            ", free=" + format_bytes(free) + ", reserve=" + format_bytes(reserve));
    }
}

void reject_native_kv_size_override(const PipelineContext& ctx) {
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "Granite native TensorRT KV cache allocates the model's complete fixed "
            "capacity; kv_cache_size_bytes is not supported");
    }
}

void reject_legacy_cache_runtime(const PipelineContext& ctx) {
    if (ctx.runtime_config == nullptr)
        return;
    bool enabled = false;
    try {
        enabled = ctx.runtime_config->get<bool>("triattention", "enabled");
    } catch (const std::exception&) {
        // A runtime without the retired schema has nothing to reject.
    }
    if (enabled)
        throw std::invalid_argument(
            "Granite native KV does not support the retired TriAttention cache");
}

KvCacheRuntimeSizing resolve_kv_cache_runtime_sizing(const PipelineContext& ctx, DType cache_dtype,
                                                     int32_t kv_dim) {
    KvCacheRuntimeSizing sizing;
    const int32_t bundle_max_rows = ctx.config.max_cache_length;
    if (ctx.config.num_layers <= 0 || kv_dim <= 0 || bundle_max_rows <= 0)
        throw std::runtime_error("Granite KV geometry must be positive");
    sizing.row_bytes = checked_multiply(
        checked_multiply(checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                          static_cast<std::uint64_t>(kv_dim)),
                         static_cast<std::uint64_t>(dtype_size(cache_dtype))),
        2);
    sizing.runtime_rows = bundle_max_rows;
    sizing.cache_bytes =
        checked_multiply(static_cast<std::uint64_t>(bundle_max_rows), sizing.row_bytes);

    reject_native_kv_size_override(ctx);
    return sizing;
}

void validate_native_kv_runtime(const PipelineContext& ctx, const TrtModule& module,
                                const GraniteKvCacheNames& kv_names, DType cache_dtype,
                                const TensorParallelRuntimeConfig& tp_config) {
    validate_native_kv_marker(ctx.config_json);
    require_native_kv_inputs(module, kv_names);
    if (!module.has_output(ctx.config.io_map.logits) ||
        module.tensor_dtype(ctx.config.io_map.logits) != DType::kFloat32) {
        throw std::runtime_error("Granite native KV requires FP32 logits output");
    }
    if (cache_dtype != DType::kFloat16 || tp_config.enabled) {
        throw std::runtime_error("Granite native KV requires FP16 and a single-GPU runtime");
    }

    if (!valid_native_cache_shape(module.tensor_shape(kv_names.cache_k.front()), ctx)) {
        throw std::runtime_error("Granite native KV requires cache shape "
                                 "[1,num_kv_heads,capacity,head_dim] with head_dim 64 or 128");
    }
}

void reject_tensor_parallel(const TensorParallelRuntimeConfig& tp_config) {
    if (tp_config.enabled)
        throw std::runtime_error("Granite native KV does not support tensor-parallel bundles");
}

void validate_decode_profile(const TrtModule& module, const std::string& token_id_name) {
    if (profile_token_length(module, token_id_name, ProfileShapeSelector::kMin) != 1 ||
        profile_token_length(module, token_id_name, ProfileShapeSelector::kOpt) != 1 ||
        profile_token_length(module, token_id_name, ProfileShapeSelector::kMax) != 1) {
        throw std::runtime_error("Granite native decode engine profile must be fixed to one token");
    }
}

int32_t validate_prefill_profile(const TrtModule& module, const std::string& token_id_name,
                                 int32_t cache_capacity) {
    const int32_t min_length =
        profile_token_length(module, token_id_name, ProfileShapeSelector::kMin);
    const int32_t opt_length =
        profile_token_length(module, token_id_name, ProfileShapeSelector::kOpt);
    const int32_t max_length =
        profile_token_length(module, token_id_name, ProfileShapeSelector::kMax);
    if (min_length != 1 || opt_length < min_length || opt_length > max_length || max_length <= 1 ||
        max_length > cache_capacity) {
        throw std::runtime_error(
            "Granite native prefill profile must be ordered, start at one token, "
            "span multiple tokens, and fit the KV capacity");
    }
    return max_length;
}

void validate_matching_kv_width(const TrtModule& module, const GraniteKvCacheNames& kv_names,
                                int32_t expected_kv_dim) {
    if (cache_row_dim_from_module(module, kv_names.cache_k.front()) != expected_kv_dim) {
        throw std::runtime_error(
            "Granite native prefill and decode engines must use the same KV row width");
    }
}

} // namespace

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        apply_text_trace_from_registry(ctx.runtime_config);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const auto& io = ctx.config.io_map;
        GraniteKvCacheNames kv_names;
        build_kv_names(ctx, io, kv_names);

        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        reject_legacy_cache_runtime(ctx);

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        reject_tensor_parallel(tp_config);

        auto decoder = load_single_profile_module(ctx, "engine_plan", nullptr, "decode");
        validate_native_kv_runtime(ctx, *decoder, kv_names, cache_dtype, tp_config);
        validate_decode_profile(*decoder, io.token_id);
        const int32_t kv_dim = cache_row_dim_from_module(*decoder, kv_names.cache_k.front());
        const auto sizing = resolve_kv_cache_runtime_sizing(ctx, cache_dtype, kv_dim);
        cudaStream_t stream = decoder->stream();

        auto prefill_module =
            load_single_profile_module(ctx, "prefill_engine_plan", stream, "prefill");
        validate_native_kv_runtime(ctx, *prefill_module, kv_names, cache_dtype, tp_config);
        validate_matching_kv_width(*prefill_module, kv_names, kv_dim);
        const int32_t prefill_max_length =
            validate_prefill_profile(*prefill_module, io.token_id, sizing.runtime_rows);

        std::vector<GraniteTextGenerationPipeline::DecoderContext> decoders;
        decoders.push_back(
            GraniteTextGenerationPipeline::DecoderContext{sizing.runtime_rows, std::move(decoder)});

        // Split prefill deserialization can consume additional execution-context
        // memory. Admit the KV allocation against the free memory that remains
        // after every engine/context needed by this pipeline has been loaded.
        admit_native_kv_allocation(ctx, sizing);
        auto state = build_inference_state(ctx, sizing, cache_dtype, kv_dim, kv_names, stream);
        log_kv_cache_sizing(ctx, sizing, state.get());

        GraniteTextGenConfig tgc;
        populate_text_gen_config(ctx, tgc, io, decoders.front(), ctx.runtime_config);
        apply_chat_template_format(ctx.bundle, tgc);
        // Native TensorRT KV engines update the shared aliased cache in place.
        tgc.prefill_max_length = prefill_max_length;
        tgc.prefill_profile_index = 0;
        tgc.prefill_log_label = "prefill engine";
        tgc.num_layers = ctx.config.num_layers;
        tgc.kv_dim = kv_dim;
        tgc.present_k_pattern = io.present_k_pattern;
        tgc.present_v_pattern = io.present_v_pattern;

        return std::make_unique<GraniteTextGenerationPipeline>(
            std::move(decoders), std::move(state), tgc, stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(prefill_module), nullptr);
    }

  private:
    static std::unique_ptr<TrtModule> load_single_profile_module(const PipelineContext& ctx,
                                                                 const std::string& section_name,
                                                                 cudaStream_t stream,
                                                                 const char* role) {
        auto* plan = find_section(ctx.bundle, section_name);
        if (plan == nullptr || plan->empty())
            throw std::runtime_error(section_name + " section is missing");
        if (ctx.backend == nullptr)
            throw std::runtime_error("No backend loaded");

        ModuleCreateOptions opts;
        opts.stream = stream;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto t0 = std::chrono::steady_clock::now();
        auto modules = ctx.backend->create_profile_modules(plan->data(), plan->size(), opts,
                                                           std::vector<int32_t>{0});
        const auto t1 = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        log_trt_load_timing(section_name.c_str(), load_ms, plan->size());
        if (modules.modules.size() != 1 || modules.modules.front().profile_idx != 0 ||
            !modules.modules.front().module) {
            throw std::runtime_error("Granite native " + std::string(role) +
                                     " engine must expose exactly profile 0");
        }
        if (modules.modules.front().module->optimization_profile_count() != 1) {
            throw std::runtime_error("Granite native " + std::string(role) +
                                     " engine must contain exactly one optimization profile");
        }
        modules.modules.front().module->set_timing_label(section_name + ":" + role);
        return std::move(modules.modules.front().module);
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

    static void build_kv_names(const PipelineContext& ctx, const IoMap& io,
                               GraniteKvCacheNames& kv_names) {
        kv_names.position_id = io.position_id;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(granite_expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(granite_expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(granite_expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(granite_expand_layer_name(io.present_v_pattern, i));
        }
    }

    static std::unique_ptr<GraniteKvCache> build_inference_state(const PipelineContext& ctx,
                                                                 const KvCacheRuntimeSizing& sizing,
                                                                 DType cache_dtype, int32_t kv_dim,
                                                                 GraniteKvCacheNames& kv_names,
                                                                 cudaStream_t stream) {
        auto state =
            std::make_unique<GraniteKvCache>(ctx.config.num_layers, sizing.runtime_rows, kv_dim,
                                             stream, cache_dtype, std::move(kv_names));
        if (!state->ok())
            throw std::runtime_error("Failed to create GraniteKvCache");
        return state;
    }

    static void log_kv_cache_sizing(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing,
                                    const GraniteKvCache* state) {
        std::cerr << "[trtmc] KV cache rows=" << sizing.runtime_rows
                  << " (bundle max=" << ctx.config.max_cache_length
                  << ", row=" << format_bytes(sizing.row_bytes)
                  << ", cache=" << format_bytes(sizing.cache_bytes) << ", state="
                  << format_bytes(static_cast<std::uint64_t>(state->device_memory_bytes())) << ")";
        std::cerr << '\n';
    }

    static void
    populate_text_gen_config(const PipelineContext& ctx, GraniteTextGenConfig& tgc, const IoMap& io,
                             const GraniteTextGenerationPipeline::DecoderContext& first_dec,
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

    static void apply_chat_template_format(const BundleFile& bundle, GraniteTextGenConfig& tgc) {
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
        tgc.chat_template_format = granite_detect_chat_template_format(chat_tpl);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_granite_plugin, DecoderPlugin,
                                       "granite_decoder_kv_cache");

} // namespace trtmc
