/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"
#include "runtime/models/gpt_neox/chat_templates.h"
#include "runtime/models/gpt_neox/pipeline.h"
#include "runtime/models/gpt_neox/tensor_names.h"
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
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct KvCacheRuntimeSizing {
    int32_t rows{0};
    std::uint64_t row_bytes{0};
    std::uint64_t cache_bytes{0};
};

int32_t dim_at(const std::vector<int64_t>& shape, std::size_t dim) {
    if (dim >= shape.size() || shape[dim] <= 0 ||
        shape[dim] > std::numeric_limits<int32_t>::max()) {
        return -1;
    }
    return static_cast<int32_t>(shape[dim]);
}

int32_t cache_row_width(const TrtModule& module, const std::string& tensor_name) {
    const auto shape = module.tensor_shape(tensor_name);
    if (shape.size() != 4) {
        throw std::runtime_error("GPT-NeoX native KV tensor '" + tensor_name +
                                 "' must have rank 4");
    }
    const int32_t heads = dim_at(shape, 1);
    const int32_t head_dim = dim_at(shape, 3);
    if (heads <= 0 || head_dim <= 0 || heads > std::numeric_limits<int32_t>::max() / head_dim) {
        throw std::runtime_error("Unable to infer GPT-NeoX native KV row width from '" +
                                 tensor_name + "'");
    }
    return heads * head_dim;
}

int32_t profile_token_length(const TrtModule& module, const std::string& token_name,
                             ProfileShapeSelector selector, const std::string& section_name) {
    const auto shape = module.input_profile_shape(token_name, 0, selector);
    if (shape.size() != 1 || shape[0] <= 0 || shape[0] > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("GPT-NeoX native KV " + section_name +
                                 " must expose a positive rank-1 token profile");
    }
    return static_cast<int32_t>(shape[0]);
}

struct TokenProfileLengths {
    int32_t min{0};
    int32_t opt{0};
    int32_t max{0};
};

TokenProfileLengths read_token_profile(const TrtModule& module, const std::string& token_name,
                                       const std::string& section_name) {
    return {
        profile_token_length(module, token_name, ProfileShapeSelector::kMin, section_name),
        profile_token_length(module, token_name, ProfileShapeSelector::kOpt, section_name),
        profile_token_length(module, token_name, ProfileShapeSelector::kMax, section_name),
    };
}

void validate_decode_profile(const TokenProfileLengths& profile) {
    if (profile.min != 1 || profile.opt != 1 || profile.max != 1) {
        throw std::runtime_error(
            "GPT-NeoX native KV engine_plan must be a single-token decode profile");
    }
}

void validate_prefill_profile(const TokenProfileLengths& profile, int32_t cache_capacity) {
    if (profile.min != 1 || profile.opt < profile.min || profile.opt > profile.max ||
        profile.max <= 1 || profile.max > cache_capacity) {
        throw std::runtime_error(
            "GPT-NeoX native KV prefill_engine_plan must be a valid multi-token profile "
            "within the fixed cache capacity");
    }
}

int32_t validate_native_profile_role(const TrtModule& module, const std::string& token_name,
                                     const std::string& section_name, bool prefill,
                                     int32_t cache_capacity) {
    if (module.optimization_profile_count() != 1 || module.profile_idx() != 0) {
        throw std::runtime_error("GPT-NeoX native KV " + section_name +
                                 " must contain exactly optimization profile 0");
    }
    const auto profile = read_token_profile(module, token_name, section_name);
    if (!prefill) {
        validate_decode_profile(profile);
        return 1;
    }
    validate_prefill_profile(profile, cache_capacity);
    return profile.max;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("GPT-NeoX native KV byte accounting overflow");
    return lhs * rhs;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream out;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    out.setf(std::ios::fixed);
    out.precision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB))
        out << static_cast<double>(bytes) / kGiB << " GiB";
    else if (bytes >= static_cast<std::uint64_t>(kMiB))
        out << static_cast<double>(bytes) / kMiB << " MiB";
    else
        out << bytes << " B";
    return out.str();
}

KvCacheRuntimeSizing resolve_kv_cache_sizing(const PipelineContext& ctx, DType cache_dtype,
                                             int32_t kv_dim) {
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "GPT-NeoX native TensorRT KV cache always allocates the model's full capacity; "
            "kv_cache_size_bytes is unsupported");
    }
    if (ctx.config.num_layers <= 0 || ctx.config.max_cache_length <= 0 || kv_dim <= 0)
        throw std::runtime_error("GPT-NeoX native KV geometry must be positive");

    KvCacheRuntimeSizing sizing;
    sizing.rows = ctx.config.max_cache_length;
    sizing.row_bytes = checked_multiply(
        checked_multiply(checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                          static_cast<std::uint64_t>(kv_dim)),
                         static_cast<std::uint64_t>(dtype_size(cache_dtype))),
        2);
    sizing.cache_bytes =
        checked_multiply(static_cast<std::uint64_t>(sizing.rows), sizing.row_bytes);
    return sizing;
}

void admit_kv_allocation(const PipelineContext& ctx, const KvCacheRuntimeSizing& sizing) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GPT-NeoX native KV CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }

    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (sizing.cache_bytes > available) {
        throw std::runtime_error(
            "GPT-NeoX native KV cache admission failed before allocation: capacity=" +
            std::to_string(ctx.config.max_cache_length) +
            " tokens, required=" + format_bytes(sizing.cache_bytes) +
            ", free=" + format_bytes(free) + ", reserve=" + format_bytes(reserve));
    }
}

void validate_native_bundle_metadata(const PipelineContext& ctx, DType cache_dtype) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error(
            "GPT-NeoX runtime only accepts native KV contract version 1 bundles");
    }
    if (extract_json_string(ctx.config_json, "decoder_engine_layout", "") != "split") {
        throw std::runtime_error(
            "GPT-NeoX native KV runtime requires split prefill/decode engines");
    }
    if (cache_dtype != DType::kFloat16)
        throw std::runtime_error("GPT-NeoX native KV runtime requires FP16");
    if (extract_json_string(ctx.config_json, "tensor_parallel_mode", "single") != "single" ||
        extract_json_int(ctx.config_json, "tensor_parallel_size", 1) != 1) {
        throw std::runtime_error("GPT-NeoX native KV runtime supports single-GPU bundles only");
    }
    if (ctx.config_json.find("\"triattention\"") != std::string::npos)
        throw std::runtime_error("GPT-NeoX native KV runtime does not support TriAttention");
}

int32_t validate_native_head_dim(const PipelineContext& ctx) {
    if (ctx.config.num_heads <= 0 || ctx.config.num_kv_heads != ctx.config.num_heads ||
        ctx.config.hidden_size <= 0 || ctx.config.hidden_size % ctx.config.num_heads != 0) {
        throw std::runtime_error("GPT-NeoX native KV runtime requires a valid dense MHA geometry");
    }

    const int32_t head_dim = ctx.config.hidden_size / ctx.config.num_heads;
    if (head_dim % 8 != 0 || head_dim > 128) {
        throw std::runtime_error(
            "GPT-NeoX native KV runtime requires head_dim to be a multiple of 8 "
            "no larger than 128");
    }
    return head_dim;
}

void validate_native_module_contract(const PipelineContext& ctx, const TrtModule& module,
                                     const GptNeoxKvCacheNames& names) {
    if (!module.has_input(names.cache_write_indices) ||
        !module.has_input(names.key_value_lengths)) {
        throw std::runtime_error(
            "GPT-NeoX native KV engine is missing cache_write_indices/key_value_lengths");
    }
    const int32_t head_dim = validate_native_head_dim(ctx);
    const std::vector<int64_t> expected_shape{
        1,
        ctx.config.num_kv_heads,
        ctx.config.max_cache_length,
        head_dim,
    };
    if (names.cache_k.empty() || module.tensor_shape(names.cache_k.front()) != expected_shape) {
        throw std::runtime_error(
            "GPT-NeoX native KV cache shape does not match the model geometry");
    }
}

int32_t validate_native_bundle(const PipelineContext& ctx, const TrtModule& module,
                               const GptNeoxKvCacheNames& names, DType cache_dtype,
                               const char* engine_role) {
    validate_native_bundle_metadata(ctx, cache_dtype);
    validate_native_module_contract(ctx, module, names);
    const bool prefill = std::string(engine_role) == "prefill";
    return validate_native_profile_role(module, ctx.config.io_map.token_id,
                                        prefill ? "prefill_engine_plan" : "engine_plan", prefill,
                                        ctx.config.max_cache_length);
}

std::unique_ptr<TrtModule> load_native_module(const PipelineContext& ctx,
                                              const std::string& section_name,
                                              cudaStream_t stream) {
    const auto* plan = find_section(ctx.bundle, section_name);
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("GPT-NeoX native KV bundle is missing " + section_name);
    if (ctx.backend == nullptr)
        throw std::runtime_error("No backend loaded");

    ModuleCreateOptions options;
    options.stream = stream;
    options.runtime_cache_path = ctx.runtime_cache_path.c_str();
    options.cuda_graphs = ctx.cuda_graphs;

    const auto start = std::chrono::steady_clock::now();
    auto modules = ctx.backend->create_profile_modules(plan->data(), plan->size(), options, {0});
    const auto end = std::chrono::steady_clock::now();
    log_trt_load_timing(section_name.c_str(),
                        std::chrono::duration<double, std::milli>(end - start).count(),
                        plan->size());
    if (modules.modules.size() != 1 || modules.modules.front().profile_idx != 0 ||
        !modules.modules.front().module) {
        throw std::runtime_error("GPT-NeoX native KV " + section_name +
                                 " failed to load exactly profile 0");
    }
    auto module = std::move(modules.modules.front().module);
    if (module->optimization_profile_count() != 1 || module->profile_idx() != 0) {
        throw std::runtime_error("GPT-NeoX native KV " + section_name +
                                 " must contain exactly one optimization profile");
    }
    module->set_timing_label(section_name);
    return module;
}

void build_kv_names(const PipelineContext& ctx, GptNeoxKvCacheNames& names) {
    const auto& io = ctx.config.io_map;
    names.position_id = io.position_id;
    for (int32_t layer = 0; layer < ctx.config.num_layers; ++layer) {
        names.cache_k.push_back(gpt_neox_expand_layer_name(io.cache_k_pattern, layer));
        names.cache_v.push_back(gpt_neox_expand_layer_name(io.cache_v_pattern, layer));
        names.present_k.push_back(gpt_neox_expand_layer_name(io.present_k_pattern, layer));
        names.present_v.push_back(gpt_neox_expand_layer_name(io.present_v_pattern, layer));
    }
}

void apply_text_trace_from_registry(const config::ConfigBundle* config) {
    if (config == nullptr)
        return;
    try {
        apply_text_trace_config_from_registry(
            config->get<std::string>("text_trace", "step_trace_path"),
            config->get<std::int32_t>("text_trace", "step_trace_start_pos"),
            config->get<std::int32_t>("text_trace", "step_trace_end_pos"),
            config->get<std::int32_t>("text_trace", "step_trace_topk"));
    } catch (const std::exception&) {
        // The namespace is optional; an absent schema leaves tracing disabled.
    }
}

void populate_text_config(const PipelineContext& ctx, GptNeoxTextGenConfig& config,
                          const TrtModule& decoder, int32_t prefill_max_length, int32_t kv_dim) {
    const auto& io = ctx.config.io_map;
    config.vocab_size = ctx.config.vocab_size;
    config.id_bos = ctx.config.id_bos;
    config.id_eos = ctx.config.id_eos;
    config.has_position_input = decoder.has_input(io.position_id);
    config.token_id_name = io.token_id;
    config.logits_output_name = io.logits;
    config.prefill_max_length = prefill_max_length;
    config.prefill_profile_index = 0;
    config.prefill_log_label = "prefill engine";
    config.num_layers = ctx.config.num_layers;
    config.kv_dim = kv_dim;
    config.present_k_pattern = io.present_k_pattern;
    config.present_v_pattern = io.present_v_pattern;
    if (ctx.runtime_config != nullptr) {
        try {
            config.disable_cuda_graph =
                ctx.runtime_config->get<bool>("runtime", "disable_cuda_graph");
            config.prefer_gpu_greedy =
                ctx.runtime_config->get<bool>("runtime", "prefer_gpu_greedy");
            config.log_runtime_stats = ctx.runtime_config->get<bool>("platform", "trt_log_stderr");
        } catch (const std::exception&) {
            // Optional runtime schemas may be absent in embedding applications.
        }
    }

    std::string chat_template;
    const auto* tokenizer_config = find_section(ctx.bundle, "tokenizer_config.json");
    if (tokenizer_config != nullptr && !tokenizer_config->empty()) {
        chat_template = extract_json_string(
            std::string(tokenizer_config->begin(), tokenizer_config->end()), "chat_template", "");
    }
    if (chat_template.empty()) {
        const auto* template_section = find_section(ctx.bundle, "chat_template.jinja");
        if (template_section != nullptr && !template_section->empty())
            chat_template.assign(template_section->begin(), template_section->end());
    }
    config.chat_template_format = gpt_neox_detect_chat_template_format(chat_template);
}

} // namespace

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        apply_text_trace_from_registry(ctx.runtime_config);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        GptNeoxKvCacheNames names;
        build_kv_names(ctx, names);
        const DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);

        auto decoder = load_native_module(ctx, "engine_plan", nullptr);
        validate_native_bundle(ctx, *decoder, names, cache_dtype, "decode");
        const int32_t kv_dim = cache_row_width(*decoder, names.cache_k.front());
        const auto sizing = resolve_kv_cache_sizing(ctx, cache_dtype, kv_dim);
        admit_kv_allocation(ctx, sizing);

        cudaStream_t stream = decoder->stream();
        auto prefill = load_native_module(ctx, "prefill_engine_plan", stream);
        const int32_t prefill_max_length =
            validate_native_bundle(ctx, *prefill, names, cache_dtype, "prefill");

        auto state = std::make_unique<GptNeoxKvCache>(ctx.config.num_layers, sizing.rows, kv_dim,
                                                      stream, cache_dtype, std::move(names));
        if (!state->ok())
            throw std::runtime_error("Failed to allocate GPT-NeoX native KV cache");
        std::cerr << "[trtmc] KV cache rows=" << sizing.rows
                  << " (row=" << format_bytes(sizing.row_bytes)
                  << ", cache=" << format_bytes(sizing.cache_bytes) << ")\n";

        GptNeoxTextGenConfig config;
        populate_text_config(ctx, config, *decoder, prefill_max_length, kv_dim);
        std::vector<GptNeoxTextGenerationPipeline::DecoderContext> decoders;
        decoders.push_back(
            GptNeoxTextGenerationPipeline::DecoderContext{sizing.rows, std::move(decoder)});
        return std::make_unique<GptNeoxTextGenerationPipeline>(
            std::move(decoders), std::move(state), std::move(config), stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(prefill));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_gpt_neox_plugin, DecoderPlugin,
                                       "gpt_neox_decoder_kv_cache");

} // namespace trtmc
