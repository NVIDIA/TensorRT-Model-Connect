/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// K2-Horizon owns this complete runtime boundary. Only the qualified dense
// BF16, single-engine, fixed native-KV contract is accepted.

#include "plugin_helpers.h"
#include "runtime/models/k2_horizon/chat_template.h"
#include "runtime/models/k2_horizon/kv_cache.h"
#include "runtime/models/k2_horizon/pipeline.h"
#include "runtime/models/k2_horizon/tensor_names.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

void require_positive(int32_t value, const char* name) {
    if (value <= 0)
        throw std::invalid_argument(std::string("K2-Horizon requires positive ") + name);
}

void validate_model_contract(const PipelineContext& ctx) {
    if (extract_json_string(ctx.config_json, "model_type", "") != "k2_horizon")
        throw std::invalid_argument("K2-Horizon runtime requires model_type='k2_horizon'");
    if (ctx.config.precision != "bf16")
        throw std::invalid_argument("K2-Horizon runtime supports BF16 bundles only");
    if (extract_json_string(ctx.config_json, "engine_backend", "") != "trt")
        throw std::invalid_argument("K2-Horizon runtime supports standard TensorRT only");
    if (ctx.config.runtime_strategy != "k2_horizon_decoder_kv_cache")
        throw std::invalid_argument("K2-Horizon runtime strategy metadata is inconsistent");
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::invalid_argument("K2-Horizon requires native KV contract version 1");
    }
    if (extract_json_bool(ctx.config_json, "dynamic_kv_cache", false))
        throw std::invalid_argument("K2-Horizon does not support dynamic KV cache");
    if (extract_json_string(ctx.config_json, "decoder_engine_layout", "single") != "single")
        throw std::invalid_argument("K2-Horizon supports one decoder engine only");
}

void validate_parallel_contract(const PipelineContext& ctx) {
    const std::string tp_mode =
        extract_json_string(ctx.config_json, "tensor_parallel_mode", "single");
    const int32_t tp_size = extract_json_int(ctx.config_json, "tensor_parallel_size", 1);
    if (tp_mode != "single" || tp_size != 1)
        throw std::invalid_argument("K2-Horizon does not support tensor-parallel bundles");
}

void validate_build_features(const std::string& config_json) {
    const auto config = nlohmann::json::parse(config_json);
    if (!config.is_object())
        throw std::invalid_argument("K2-Horizon config.json root must be an object");

    const auto quantization = config.find("quantization");
    if (quantization != config.end() && !quantization->is_null())
        throw std::invalid_argument("K2-Horizon does not support quantized runtime bundles");

    const auto fp32_layers = config.find("fp32_layers");
    if (fp32_layers == config.end())
        return;
    if (!fp32_layers->is_array() || !fp32_layers->empty())
        throw std::invalid_argument("K2-Horizon does not support mixed-FP32 runtime bundles");
}

void validate_bundle_sections(const PipelineContext& ctx) {
    if (ctx.kv_cache_size_bytes != 0)
        throw std::invalid_argument("K2-Horizon does not support runtime KV-size overrides");
    if (find_section(ctx.bundle, "kernel_manifest.json") != nullptr)
        throw std::invalid_argument("K2-Horizon does not support embedded FFI kernels");
    if (find_section(ctx.bundle, "kernel_slots.json") != nullptr)
        throw std::invalid_argument("K2-Horizon does not support external FFI kernel slots");
    if (find_section(ctx.bundle, "prefill_engine_plan") != nullptr)
        throw std::invalid_argument("K2-Horizon does not support a split prefill engine");
}

void validate_base_dimensions(const PipelineContext& ctx) {
    require_positive(ctx.config.vocab_size, "vocab_size");
    require_positive(ctx.config.num_layers, "num_layers");
    require_positive(ctx.config.num_heads, "num_heads");
    require_positive(ctx.config.num_kv_heads, "num_kv_heads");
    require_positive(ctx.config.max_cache_length, "max_cache_length");
    if (ctx.config.head_dim != 128)
        throw std::invalid_argument("K2-Horizon native attention requires head_dim=128");
    if (ctx.config.num_heads % ctx.config.num_kv_heads != 0)
        throw std::invalid_argument("K2-Horizon num_heads must be divisible by num_kv_heads");
}

void validate_resource_bounds(const PipelineContext& ctx) {
    constexpr int32_t max_layers = 4096;
    constexpr int32_t max_heads = 65536;
    constexpr int32_t max_cache_length = 4 * 1024 * 1024;
    constexpr int32_t max_vocab_size = 10 * 1024 * 1024;
    if (ctx.config.num_layers > max_layers || ctx.config.num_heads > max_heads ||
        ctx.config.num_kv_heads > max_heads || ctx.config.max_cache_length > max_cache_length ||
        ctx.config.vocab_size > max_vocab_size) {
        throw std::invalid_argument("K2-Horizon bundle dimensions exceed runtime safety bounds");
    }
}

void reject_unqualified_bundle(const PipelineContext& ctx) {
    validate_model_contract(ctx);
    validate_parallel_contract(ctx);
    validate_build_features(ctx.config_json);
    validate_bundle_sections(ctx);
    validate_base_dimensions(ctx);
    validate_resource_bounds(ctx);
}

void require_input(const TrtModule& module, const std::string& name, DType dtype,
                   const std::vector<int64_t>& shape) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != shape) {
        throw std::invalid_argument("K2-Horizon engine input contract mismatch for '" + name + "'");
    }
}

std::set<std::string> tensor_names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> names;
    for (const auto& tensor : tensors)
        names.insert(tensor.name);
    return names;
}

void validate_io_inventory(const PipelineContext& ctx, const TrtModule& module,
                           const K2HorizonKvCacheNames& cache_names) {
    std::set<std::string> expected_inputs{ctx.config.io_map.token_id, ctx.config.io_map.position_id,
                                          cache_names.cache_write_indices,
                                          cache_names.key_value_lengths};
    expected_inputs.insert(cache_names.cache_k.begin(), cache_names.cache_k.end());
    expected_inputs.insert(cache_names.cache_v.begin(), cache_names.cache_v.end());
    const std::size_t input_count = 4 + 2 * static_cast<std::size_t>(ctx.config.num_layers);
    if (expected_inputs.size() != input_count ||
        tensor_names(module.input_info()) != expected_inputs)
        throw std::invalid_argument("K2-Horizon engine input inventory is not exact");

    std::set<std::string> expected_outputs{ctx.config.io_map.logits};
    expected_outputs.insert(cache_names.present_k.begin(), cache_names.present_k.end());
    expected_outputs.insert(cache_names.present_v.begin(), cache_names.present_v.end());
    const std::size_t output_count = 1 + 2 * static_cast<std::size_t>(ctx.config.num_layers);
    if (expected_outputs.size() != output_count ||
        tensor_names(module.output_info()) != expected_outputs) {
        throw std::invalid_argument("K2-Horizon engine output inventory is not exact");
    }
}

void validate_engine_contract(const PipelineContext& ctx, const TrtModule& module,
                              const K2HorizonKvCacheNames& cache_names) {
    if (module.optimization_profile_count() != 1)
        throw std::invalid_argument("K2-Horizon requires exactly one optimization profile");

    const auto& io = ctx.config.io_map;
    require_input(module, io.token_id, DType::kInt32, {1});
    require_input(module, io.position_id, DType::kInt32, {1});
    require_input(module, "cache_write_indices", DType::kInt32, {1});
    require_input(module, "key_value_lengths", DType::kInt32, {1});
    if (!module.has_output(io.logits) || module.tensor_dtype(io.logits) != DType::kFloat32 ||
        module.tensor_shape(io.logits) !=
            std::vector<int64_t>{1, static_cast<int64_t>(ctx.config.vocab_size)}) {
        throw std::invalid_argument("K2-Horizon engine logits must be float32 [1,vocab_size]");
    }
    validate_io_inventory(ctx, module, cache_names);
}

K2HorizonKvCacheNames make_cache_names(const PipelineContext& ctx) {
    K2HorizonKvCacheNames names;
    names.position_id = ctx.config.io_map.position_id;
    for (int32_t layer = 0; layer < ctx.config.num_layers; ++layer) {
        names.cache_k.push_back(
            k2_horizon_expand_layer_name(ctx.config.io_map.cache_k_pattern, layer));
        names.cache_v.push_back(
            k2_horizon_expand_layer_name(ctx.config.io_map.cache_v_pattern, layer));
        names.present_k.push_back(
            k2_horizon_expand_layer_name(ctx.config.io_map.present_k_pattern, layer));
        names.present_v.push_back(
            k2_horizon_expand_layer_name(ctx.config.io_map.present_v_pattern, layer));
    }
    return names;
}

bool runtime_flag(const config::ConfigBundle* config, const char* name_space, const char* key,
                  bool fallback) {
    if (config == nullptr)
        return fallback;
    try {
        return config->get<bool>(name_space, key);
    } catch (const std::exception&) {
        return fallback;
    }
}

std::vector<int32_t> normalized_eos_token_ids(std::vector<int32_t> token_ids,
                                              int32_t fallback_token_id, int32_t vocab_size) {
    if (token_ids.empty() && fallback_token_id >= 0)
        token_ids.push_back(fallback_token_id);

    std::vector<int32_t> normalized;
    normalized.reserve(token_ids.size());
    for (int32_t token_id : token_ids) {
        if (token_id < 0 || token_id >= vocab_size)
            throw std::invalid_argument("K2-Horizon EOS token is outside the model vocabulary");
        if (std::find(normalized.begin(), normalized.end(), token_id) == normalized.end())
            normalized.push_back(token_id);
    }
    return normalized;
}

std::string load_chat_template_format(const PipelineContext& ctx) {
    const auto* section = find_section(ctx.bundle, "chat_template.jinja");
    if (section == nullptr || section->empty()) {
        throw std::invalid_argument(
            "K2-Horizon bundle is missing the required publisher chat template");
    }
    if (!ctx.config.tokenizer_add_special_tokens_present ||
        !ctx.config.tokenizer_add_special_tokens) {
        throw std::invalid_argument("K2-Horizon chat requires tokenizer_add_special_tokens=1");
    }
    return k2_horizon_detect_chat_template_format(std::string(section->begin(), section->end()));
}

void validate_chat_tokenizer_contract(const PipelineContext& ctx, const ITokenizer& tokenizer,
                                      const std::string& chat_template_format) {
    constexpr int32_t bos_id = 0;
    constexpr int32_t im_start_id = 250018;
    constexpr int32_t im_end_id = 250019;
    constexpr int32_t think_id = 250029;
    if (ctx.config.id_bos != bos_id || tokenizer.id_for_token("<|ifm|begin_of_text|>") != bos_id) {
        throw std::invalid_argument("K2-Horizon chat BOS token metadata is inconsistent");
    }
    if (tokenizer.id_for_token("<|ifm|im_start|>") != im_start_id ||
        tokenizer.id_for_token("<|ifm|im_end|>") != im_end_id ||
        tokenizer.id_for_token("<ifm|think>") != think_id) {
        throw std::invalid_argument("K2-Horizon chat protocol token IDs are inconsistent");
    }
    const auto framed_probe =
        tokenizer.encode(k2_horizon_apply_chat_template(chat_template_format, "", "high"));
    const std::vector<int32_t> expected_probe{bos_id,      im_start_id, 2672, 200,      im_end_id,
                                              im_start_id, 142036,      200,  think_id, 200};
    if (std::any_of(expected_probe.begin(), expected_probe.end(), [&](int32_t token_id) {
            return token_id < 0 || token_id >= ctx.config.vocab_size;
        })) {
        throw std::invalid_argument(
            "K2-Horizon chat framing token is outside the model vocabulary");
    }
    if (framed_probe != expected_probe) {
        throw std::invalid_argument(
            "K2-Horizon native tokenizer does not reproduce the pinned chat framing");
    }
}

} // namespace

class K2HorizonDecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        reject_unqualified_bundle(ctx);
        auto eos_token_ids = normalized_eos_token_ids(ctx.config.id_eos_ids, ctx.config.id_eos,
                                                      ctx.config.vocab_size);
        k2_horizon_validate_chat_eos_token_ids(eos_token_ids);
        const std::string chat_template_format = load_chat_template_format(ctx);
        auto tokenizer = create_k2_horizon_bpe_tokenizer(ctx.bundle);
        validate_chat_tokenizer_contract(ctx, *tokenizer, chat_template_format);

        const bool prefer_gpu_greedy =
            runtime_flag(ctx.runtime_config, "runtime", "prefer_gpu_greedy", false);
        if (prefer_gpu_greedy)
            throw std::invalid_argument("K2-Horizon does not support GPU sampling");
        const bool disable_cuda_graph =
            runtime_flag(ctx.runtime_config, "runtime", "disable_cuda_graph", false);
        const bool enable_cuda_graph = ctx.cuda_graphs && !disable_cuda_graph;

        ModuleCreateOptions options;
        options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        options.cuda_graphs = enable_cuda_graph;
        auto decoder = load_k2_horizon_engine_plan(ctx.backend, ctx.bundle, options);
        auto cache_names = make_cache_names(ctx);
        validate_engine_contract(ctx, *decoder, cache_names);

        if (ctx.config.num_kv_heads > std::numeric_limits<int32_t>::max() / 128)
            throw std::overflow_error("K2-Horizon KV width overflow");
        const int32_t kv_dim = ctx.config.num_kv_heads * 128;
        auto cache =
            std::make_unique<K2HorizonKvCache>(ctx.config.num_layers, ctx.config.max_cache_length,
                                               kv_dim, decoder->stream(), std::move(cache_names));
        if (!cache->ok())
            throw std::runtime_error("K2-Horizon native KV cache allocation failed");
        cache->bind_to(*decoder);

        K2HorizonTextGenConfig generation;
        generation.vocab_size = ctx.config.vocab_size;
        generation.eos_token_ids = std::move(eos_token_ids);
        generation.token_id_name = ctx.config.io_map.token_id;
        generation.logits_output_name = ctx.config.io_map.logits;
        generation.chat_template_format = chat_template_format;
        generation.enable_cuda_graph = enable_cuda_graph;
        generation.log_runtime_stats =
            runtime_flag(ctx.runtime_config, "platform", "trt_log_stderr", false);
        generation.emit_prompt_token_ids =
            runtime_flag(ctx.runtime_config, "k2_horizon", "emit_prompt_token_ids", false);

        return std::make_unique<K2HorizonTextGenerationPipeline>(
            std::move(decoder), std::move(cache), std::move(generation), std::move(tokenizer),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_k2_horizon_plugin, K2HorizonDecoderPlugin,
                                       "k2_horizon_decoder_kv_cache");

} // namespace trtmc
