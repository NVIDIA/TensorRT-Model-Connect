/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon/runtime/kv_cache.h"
#include "families/k2_horizon/runtime/pipeline.h"
#include "families/k2_horizon/runtime/plugin_helpers.h"
#include "families/k2_horizon/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::k2_horizon {
namespace {

struct RuntimeConfig {
    std::int32_t vocab_size;
    std::int32_t hidden_size;
    std::int32_t intermediate_size;
    std::int32_t num_layers;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    std::int32_t layernorm_num_groups;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t pad_token_id;
    std::int32_t max_cache_length;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("K2-Horizon runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("K2-Horizon runtime.json has invalid '") + name + "'");
    }
}

std::vector<std::int32_t> require_eos_ids(const nlohmann::json& json) {
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return {value.get<std::int32_t>()};
    if (value.is_array() && !value.empty())
        return value.get<std::vector<std::int32_t>>();
    throw std::runtime_error("K2-Horizon runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(require_text_section(bundle, "runtime.json"));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("K2-Horizon invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object() || json.size() != 24)
        throw std::runtime_error("K2-Horizon runtime.json has an unexpected field set");
    if (require_value<std::string>(json, "model_type") != "k2_horizon" ||
        require_value<std::string>(json, "architecture") != "K2HorizonForCausalLM" ||
        require_value<std::string>(json, "precision") != "bf16" ||
        require_value<std::string>(json, "decoder_engine_layout") != "single" ||
        require_value<std::int32_t>(json, "tensor_parallel_size") != 1 ||
        require_value<std::string>(json, "tensor_parallel_mode") != "single" ||
        !require_value<bool>(json, "native_kv_cache") ||
        require_value<std::int32_t>(json, "native_kv_contract_version") != 1 ||
        require_value<bool>(json, "tie_word_embeddings")) {
        throw std::runtime_error("K2-Horizon runtime.json violates the qualified contract");
    }

    RuntimeConfig config{
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "intermediate_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "max_position_embeddings"),
        require_value<float>(json, "rms_norm_eps"),
        require_value<float>(json, "rope_theta"),
        require_value<std::int32_t>(json, "layernorm_num_groups"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_ids(json),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "max_cache_length"),
    };
    constexpr std::int32_t max_layers = 4096;
    constexpr std::int32_t max_heads = 65536;
    constexpr std::int32_t max_cache_length = 4 * 1024 * 1024;
    constexpr std::int32_t max_vocab_size = 10 * 1024 * 1024;
    const auto attention_width = static_cast<std::int64_t>(config.num_heads) * config.head_dim;
    if (config.vocab_size <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_layers <= 0 || config.num_heads <= 0 || config.num_kv_heads <= 0 ||
        config.head_dim != 128 || config.max_position_embeddings <= 0 ||
        config.rms_norm_eps <= 0.0F || config.rope_theta <= 0.0F ||
        config.layernorm_num_groups != 4 ||
        static_cast<std::int64_t>(config.hidden_size) != attention_width ||
        config.num_heads % config.num_kv_heads != 0 || config.max_cache_length <= 0 ||
        config.max_cache_length > config.max_position_embeddings || config.eos_token_ids.empty()) {
        throw std::runtime_error("K2-Horizon runtime.json contains invalid model geometry");
    }
    if (config.num_layers > max_layers || config.num_heads > max_heads ||
        config.num_kv_heads > max_heads || config.max_cache_length > max_cache_length ||
        config.vocab_size > max_vocab_size) {
        throw std::runtime_error("K2-Horizon runtime.json exceeds runtime safety bounds");
    }
    for (const auto token : config.eos_token_ids) {
        if (token < 0 || token >= config.vocab_size)
            throw std::runtime_error("K2-Horizon runtime.json has out-of-range EOS token");
    }
    return config;
}

K2HorizonKvCacheNames make_cache_names(std::int32_t num_layers) {
    K2HorizonKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(k2_horizon_layer_tensor_name("cache_k", layer));
        names.cache_v.push_back(k2_horizon_layer_tensor_name("cache_v", layer));
        names.present_k.push_back(k2_horizon_layer_tensor_name("present_k", layer));
        names.present_v.push_back(k2_horizon_layer_tensor_name("present_v", layer));
    }
    return names;
}

void require_input(const ITrtModule& module, const std::string& name, DType dtype,
                   const std::vector<std::int64_t>& shape) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != shape) {
        throw std::runtime_error("K2-Horizon engine input contract mismatch for '" + name + "'");
    }
}

std::set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> result;
    for (const auto& tensor : tensors)
        result.insert(tensor.name);
    return result;
}

void validate_engine(const RuntimeConfig& config, const ITrtModule& module,
                     const K2HorizonKvCacheNames& cache_names) {
    if (module.optimization_profile_count() != 1)
        throw std::runtime_error("K2-Horizon requires exactly one optimization profile");
    require_input(module, "token_id", DType::kInt32, {1});
    require_input(module, "position_id", DType::kInt32, {1});
    require_input(module, "cache_write_indices", DType::kInt32, {1});
    require_input(module, "key_value_lengths", DType::kInt32, {1});
    if (!module.has_output("logits") || module.tensor_dtype("logits") != DType::kFloat32 ||
        module.tensor_shape("logits") !=
            std::vector<std::int64_t>{1, static_cast<std::int64_t>(config.vocab_size)}) {
        throw std::runtime_error("K2-Horizon engine logits must be float32 [1,vocab_size]");
    }

    std::set<std::string> expected_inputs{"token_id", "position_id", "cache_write_indices",
                                          "key_value_lengths"};
    expected_inputs.insert(cache_names.cache_k.begin(), cache_names.cache_k.end());
    expected_inputs.insert(cache_names.cache_v.begin(), cache_names.cache_v.end());
    std::set<std::string> expected_outputs{"logits"};
    expected_outputs.insert(cache_names.present_k.begin(), cache_names.present_k.end());
    expected_outputs.insert(cache_names.present_v.begin(), cache_names.present_v.end());
    if (names(module.input_info()) != expected_inputs ||
        names(module.output_info()) != expected_outputs) {
        throw std::runtime_error("K2-Horizon engine tensor inventory is not exact");
    }
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (std::string(context.backend.name()) != "trt")
        throw std::runtime_error("K2-Horizon requires the TensorRT backend");
    for (const char* section : {"prefill.plan", "kernel_manifest.json", "kernel_slots.json"}) {
        if (context.reader.find_section(section) != nullptr)
            throw std::runtime_error(std::string("K2-Horizon rejects bundle section '") + section +
                                     "'");
    }
    const RuntimeConfig config = parse_runtime_config(context.reader);
    auto decoder = load_engine(context.backend, require_section(context.reader, "engine.plan"));
    auto cache_names = make_cache_names(config.num_layers);
    validate_engine(config, *decoder, cache_names);
    if (config.num_kv_heads > std::numeric_limits<std::int32_t>::max() / config.head_dim)
        throw std::overflow_error("K2-Horizon KV width overflow");
    auto cache = std::make_unique<K2HorizonKvCache>(config.num_layers, config.max_cache_length,
                                                    config.num_kv_heads * config.head_dim,
                                                    decoder->stream(), std::move(cache_names));
    if (!cache->ok())
        throw std::runtime_error("K2-Horizon native KV cache allocation failed");
    cache->bind_to(*decoder);

    K2HorizonTextGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.eos_token_ids = config.eos_token_ids;
    return new K2HorizonTextGenerationPipeline(std::move(decoder), std::move(cache),
                                               std::move(generation),
                                               create_tokenizer(context.reader));
}

} // namespace trtmc::k2_horizon

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::k2_horizon::create(context);
}
