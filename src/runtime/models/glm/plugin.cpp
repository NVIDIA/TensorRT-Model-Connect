/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "plugin_helpers.h"
#include "runtime/models/glm/chat_templates.h"
#include "runtime/models/glm/pipeline.h"
#include "runtime/models/glm/tensor_names.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
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

struct NativeKvSizing {
    int32_t capacity{0};
    std::uint64_t row_bytes{0};
    std::uint64_t cache_bytes{0};
};

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("GLM native KV byte accounting overflow");
    return lhs * rhs;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream output;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    output.setf(std::ios::fixed);
    output.precision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        output << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        output << static_cast<double>(bytes) / kMiB << " MiB";
    } else {
        output.unsetf(std::ios::floatfield);
        output << bytes << " B";
    }
    return output.str();
}

int32_t dimension(const std::vector<int64_t>& shape, std::size_t index) {
    if (index >= shape.size() || shape[index] <= 0 ||
        shape[index] > std::numeric_limits<int32_t>::max()) {
        return -1;
    }
    return static_cast<int32_t>(shape[index]);
}

bool runtime_triattention_requested(const config::ConfigBundle* config) {
    if (config == nullptr)
        return false;
    try {
        return config->get<bool>("triattention", "enabled");
    } catch (const std::exception&) {
        return false;
    }
}

void require_native_bundle_contract(const PipelineContext& context) {
    const bool native = extract_json_bool(context.config_json, "native_kv_cache", false);
    const int32_t version = extract_json_int(context.config_json, "native_kv_contract_version", 0);
    if (!native || version != 1) {
        throw std::runtime_error(
            "GLM runtime rejects legacy or unmarked bundles; rebuild with the native KV builder");
    }
    if (context.config.precision != "bf16")
        throw std::runtime_error("GLM native KV runtime requires BF16");
    if (extract_json_string(context.config_json, "tensor_parallel_mode", "single") != "single" ||
        extract_json_int(context.config_json, "tensor_parallel_size", 1) != 1) {
        throw std::runtime_error("GLM native KV runtime supports one GPU only");
    }
    if (context.config_json.find("\"triattention\"") != std::string::npos ||
        runtime_triattention_requested(context.runtime_config)) {
        throw std::runtime_error("GLM native KV runtime does not support TriAttention");
    }
    if (context.config_json.find("\"dynamic_kv_profile_rows\"") != std::string::npos) {
        throw std::runtime_error("GLM native KV runtime rejects legacy dynamic-row bundles");
    }
    if (context.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "GLM native KV allocates the model's complete context; kv_cache_size_bytes is invalid");
    }
}

int32_t resolved_head_dim(const BaseConfig& config) {
    if (config.head_dim > 0)
        return config.head_dim;
    if (config.num_heads <= 0 || config.hidden_size <= 0 ||
        config.hidden_size % config.num_heads != 0) {
        return -1;
    }
    return config.hidden_size / config.num_heads;
}

NativeKvSizing resolve_native_kv_sizing(const PipelineContext& context) {
    const int32_t head_dim = resolved_head_dim(context.config);
    if (context.config.num_layers <= 0 || context.config.num_kv_heads <= 0 ||
        context.config.max_cache_length <= 0 || head_dim != 128) {
        throw std::runtime_error("GLM native KV bundle has invalid model geometry");
    }

    NativeKvSizing sizing;
    sizing.capacity = context.config.max_cache_length;
    const auto kv_dim = checked_multiply(static_cast<std::uint64_t>(context.config.num_kv_heads),
                                         static_cast<std::uint64_t>(head_dim));
    sizing.row_bytes = checked_multiply(
        checked_multiply(static_cast<std::uint64_t>(context.config.num_layers), kv_dim),
        2U * static_cast<std::uint64_t>(dtype_size(DType::kBFloat16)));
    sizing.cache_bytes =
        checked_multiply(static_cast<std::uint64_t>(sizing.capacity), sizing.row_bytes);
    return sizing;
}

void admit_native_kv_allocation(const PipelineContext& context, const NativeKvSizing& sizing) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GLM native KV CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }

    constexpr std::uint64_t kTwoGiB = 2ULL << 30;
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max(kTwoGiB, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (sizing.cache_bytes > available) {
        throw std::runtime_error("GLM native KV admission failed before allocation: capacity=" +
                                 std::to_string(context.config.max_cache_length) +
                                 " tokens, required=" + format_bytes(sizing.cache_bytes) +
                                 ", free=" + format_bytes(free) +
                                 ", reserve=" + format_bytes(reserve));
    }
}

GlmKvCacheNames build_kv_names(const PipelineContext& context, const IoMap& io) {
    GlmKvCacheNames names;
    names.position_id = io.position_id;
    names.attention_mask = io.attention_mask;
    for (int32_t layer = 0; layer < context.config.num_layers; ++layer) {
        names.cache_k.push_back(glm_expand_layer_name(io.cache_k_pattern, layer));
        names.cache_v.push_back(glm_expand_layer_name(io.cache_v_pattern, layer));
        names.present_k.push_back(glm_expand_layer_name(io.present_k_pattern, layer));
        names.present_v.push_back(glm_expand_layer_name(io.present_v_pattern, layer));
    }
    return names;
}

void validate_native_scalar(const TrtModule& module, const std::string& name) {
    if (module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("GLM native KV scalar input '" + name + "' must be int32 [1]");
    }
}

void validate_native_engine_io(const TrtModule& module, const GlmKvCacheNames& names) {
    if (module.optimization_profile_count() != 1)
        throw std::runtime_error("GLM native split engines require exactly one profile each");
    if (!module.has_input(names.cache_write_indices) ||
        !module.has_input(names.key_value_lengths) || module.has_input(names.attention_mask)) {
        throw std::runtime_error("GLM engine does not implement the native KV scalar contract");
    }
    for (const auto& scalar : {names.cache_write_indices, names.key_value_lengths}) {
        validate_native_scalar(module, scalar);
    }
}

void validate_native_name_count(const GlmKvCacheNames& names, std::size_t expected_layers) {
    if (names.cache_k.size() != expected_layers || names.cache_v.size() != expected_layers ||
        names.present_k.size() != expected_layers || names.present_v.size() != expected_layers) {
        throw std::runtime_error("GLM native KV tensor names do not match num_layers");
    }
}

void validate_native_cache_pair(const TrtModule& module, const std::string& cache_name,
                                const std::string& present_name,
                                const std::vector<int64_t>& expected_shape) {
    if (!module.has_input(cache_name) || !module.has_output(present_name) ||
        module.tensor_shape(cache_name) != expected_shape ||
        module.tensor_shape(present_name) != expected_shape ||
        module.tensor_dtype(cache_name) != DType::kBFloat16 ||
        module.tensor_dtype(present_name) != DType::kBFloat16) {
        throw std::runtime_error("GLM native KV cache/present tensors must be BF16 "
                                 "[1,num_kv_heads,capacity,128]");
    }
}

void validate_native_cache_tensors(const PipelineContext& context, const TrtModule& module,
                                   const GlmKvCacheNames& names) {
    const int32_t head_dim = resolved_head_dim(context.config);
    const std::vector<int64_t> expected_shape{1, context.config.num_kv_heads,
                                              context.config.max_cache_length, head_dim};
    const auto expected_layers = static_cast<std::size_t>(context.config.num_layers);
    validate_native_name_count(names, expected_layers);
    for (std::size_t layer = 0; layer < expected_layers; ++layer) {
        validate_native_cache_pair(module, names.cache_k[layer], names.present_k[layer],
                                   expected_shape);
        validate_native_cache_pair(module, names.cache_v[layer], names.present_v[layer],
                                   expected_shape);
    }
}

int32_t validate_native_token_profiles(const PipelineContext& context, const TrtModule& module,
                                       bool prefill) {
    const auto token_profile =
        module.input_profile_shape(context.config.io_map.token_id, 0, ProfileShapeSelector::kMax);
    const auto position_profile = module.input_profile_shape(context.config.io_map.position_id, 0,
                                                             ProfileShapeSelector::kMax);
    const int32_t token_max = dimension(token_profile, 0);
    const int32_t position_max = dimension(position_profile, 0);
    if (token_max <= 0 || token_max != position_max) {
        throw std::runtime_error("GLM token and position profiles must have equal positive maxima");
    }
    if (prefill && token_max <= 1) {
        throw std::runtime_error("GLM prefill engine requires a multi-token profile");
    }
    if (!prefill && token_max != 1) {
        throw std::runtime_error("GLM decode engine requires a fixed one-token profile");
    }
    return token_max;
}

int32_t validate_native_engine(const PipelineContext& context, TrtModule& module,
                               const GlmKvCacheNames& names, bool prefill) {
    validate_native_engine_io(module, names);
    validate_native_cache_tensors(context, module, names);
    return validate_native_token_profiles(context, module, prefill);
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
        // Missing schema leaves tracing disabled.
    }
}

void apply_chat_template(const BundleFile& bundle, GlmTextGenConfig& config) {
    std::string chat_template;
    if (const auto* tokenizer_config = find_section(bundle, "tokenizer_config.json");
        tokenizer_config != nullptr && !tokenizer_config->empty()) {
        const std::string json(tokenizer_config->begin(), tokenizer_config->end());
        chat_template = extract_json_string(json, "chat_template", "");
    }
    if (chat_template.empty()) {
        if (const auto* section = find_section(bundle, "chat_template.jinja");
            section != nullptr && !section->empty()) {
            chat_template.assign(section->begin(), section->end());
        }
    }
    config.chat_template_format = glm_detect_chat_template_format(chat_template);
}

void populate_runtime_options(const config::ConfigBundle* runtime, GlmTextGenConfig& config) {
    if (runtime == nullptr)
        return;
    try {
        config.disable_cuda_graph = runtime->get<bool>("runtime", "disable_cuda_graph");
        config.prefer_gpu_greedy = runtime->get<bool>("runtime", "prefer_gpu_greedy");
        config.log_runtime_stats = runtime->get<bool>("platform", "trt_log_stderr");
    } catch (const std::exception&) {
        // Missing schemas leave stable defaults.
    }
}

} // namespace

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        require_native_bundle_contract(context);
        apply_text_trace_from_registry(context.runtime_config);

        const auto& io = context.config.io_map;
        const auto names = build_kv_names(context, io);
        const auto sizing = resolve_native_kv_sizing(context);

        ModuleCreateOptions options;
        options.runtime_cache_path = context.runtime_cache_path.c_str();
        options.cuda_graphs = context.cuda_graphs;
        auto decode =
            load_trt_module_from_plan(context.backend, find_section(context.bundle, "engine_plan"),
                                      "engine_plan", options)
                .module;
        const cudaStream_t stream = decode->stream();

        options.stream = stream;
        auto prefill = load_trt_module_from_plan(
                           context.backend, find_section(context.bundle, "prefill_engine_plan"),
                           "prefill_engine_plan", options)
                           .module;

        (void)validate_native_engine(context, *decode, names, false);
        const int32_t prefill_max = validate_native_engine(context, *prefill, names, true);
        admit_native_kv_allocation(context, sizing);

        const int32_t kv_dim = context.config.num_kv_heads * resolved_head_dim(context.config);
        auto state = std::make_unique<GlmKvCache>(context.config.num_layers, sizing.capacity,
                                                  kv_dim, stream, DType::kBFloat16, names);
        if (!state->ok())
            throw std::runtime_error("GLM native KV cache allocation failed");

        std::cerr << "[trtmc] GLM native KV capacity=" << sizing.capacity
                  << " tokens, cache=" << format_bytes(sizing.cache_bytes) << '\n';

        GlmTextGenConfig generation;
        generation.vocab_size = context.config.vocab_size;
        generation.id_bos = context.config.id_bos;
        generation.id_eos = context.config.id_eos;
        generation.token_id_name = io.token_id;
        generation.logits_output_name = io.logits;
        generation.present_k_pattern = io.present_k_pattern;
        generation.present_v_pattern = io.present_v_pattern;
        generation.prefill_max_length = prefill_max;
        generation.prefill_log_label = "native prefill engine";
        generation.num_layers = context.config.num_layers;
        populate_runtime_options(context.runtime_config, generation);
        apply_chat_template(context.bundle, generation);

        auto tokenizer = create_tokenizer_from_bundle(context.bundle);
        return std::make_unique<GlmTextGenerationPipeline>(
            std::move(decode), std::move(prefill), std::move(state), std::move(generation), stream,
            std::move(tokenizer), context.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_glm_plugin, DecoderPlugin, "glm_decoder_kv_cache");

} // namespace trtmc
