/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/internlm/runtime/chat_templates.h"
#include "families/internlm/runtime/distributed_runtime.h"
#include "families/internlm/runtime/kv_cache.h"
#include "families/internlm/runtime/pipeline.h"
#include "families/internlm/runtime/plugin_helpers.h"
#include "families/internlm/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::internlm {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::int32_t eos_token_id;
    std::int32_t pad_token_id;
    std::int32_t max_cache_length;
    std::int32_t tensor_parallel_size;
    std::string tensor_parallel_mode;
    std::string precision;
    std::string decoder_engine_layout;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("internlm runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("internlm runtime.json has invalid '") + name + "'");
    }
}

std::int32_t require_eos_token(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("internlm runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return value.get<std::int32_t>();
    if (value.is_array() && !value.empty() && value.front().is_number_integer())
        return value.front().get<std::int32_t>();
    throw std::runtime_error("internlm runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("internlm invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("internlm runtime.json must be an object");
    if (json.size() != 14)
        throw std::runtime_error("internlm runtime.json has an unexpected field set");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_token(json),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::int32_t>(json, "tensor_parallel_size"),
        require_value<std::string>(json, "tensor_parallel_mode"),
        require_value<std::string>(json, "precision"),
        require_value<std::string>(json, "decoder_engine_layout"),
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.max_cache_length <= 0 || config.tensor_parallel_size <= 0 ||
        config.num_key_value_heads * config.head_dim > config.hidden_size) {
        throw std::runtime_error("internlm runtime.json contains invalid dimensions");
    }
    if (config.num_key_value_heads % config.tensor_parallel_size != 0)
        throw std::runtime_error("internlm KV heads must be divisible by tensor parallel size");
    if (config.precision != "fp16" && config.precision != "bf16" && config.precision != "fp32") {
        throw std::runtime_error("internlm runtime.json contains invalid precision");
    }
    if (config.decoder_engine_layout != "split" && config.decoder_engine_layout != "dual_profile") {
        throw std::runtime_error("internlm runtime.json contains invalid decoder_engine_layout");
    }
    if (config.decoder_engine_layout == "split" && config.tensor_parallel_size != 1)
        throw std::runtime_error("internlm split engines require tensor_parallel_size=1");
    const std::string expected_mode =
        config.tensor_parallel_size > 1 ? "tensor_parallel" : "single";
    if (config.tensor_parallel_mode != expected_mode)
        throw std::runtime_error("internlm runtime.json has inconsistent tensor_parallel_mode");
    return config;
}

DType cache_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

InternlmKvCacheNames make_kv_names(std::int32_t num_layers) {
    InternlmKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(internlm_layer_tensor_name("cache_k", layer));
        names.cache_v.push_back(internlm_layer_tensor_name("cache_v", layer));
        names.present_k.push_back(internlm_layer_tensor_name("present_k", layer));
        names.present_v.push_back(internlm_layer_tensor_name("present_v", layer));
    }
    return names;
}

std::string chat_template(const BundleReader& bundle) {
    const auto* section = bundle.find_section("chat_template.jinja");
    if (section == nullptr || section->length == 0)
        return {};
    const auto data = bundle.read_section("chat_template.jinja");
    return {data.begin(), data.end()};
}

struct DecoderModules {
    std::unique_ptr<ITrtModule> prefill;
    std::unique_ptr<ITrtModule> decode;
    std::shared_ptr<void> distributed_owner;
};

DecoderModules load_modules(const FamilyContext& context, const RuntimeConfig& config) {
    DistributedRuntimeGroup group = initialize_tensor_parallel_group(config.tensor_parallel_size);
    DecoderModules modules;
    modules.distributed_owner = group.owner;
    if (config.decoder_engine_layout == "split") {
        modules.decode = load_engine(
            context.backend, require_section(context.reader, "engine.plan"), "internlm decoder");
        modules.prefill = load_engine(
            context.backend, require_section(context.reader, "prefill.plan"), "internlm prefill");
        return modules;
    }

    ModuleCreateOptions options;
    if (config.tensor_parallel_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string section = config.tensor_parallel_size > 1
                                    ? "engine.rank" + std::to_string(group.rank) + ".plan"
                                    : std::string("engine.plan");
    const auto& plan = require_section(context.reader, section);
    auto dual = context.backend.create_dual_profile_modules(plan.data(), plan.size(), options);
    if (dual.decode == nullptr || !dual.decode->ok() || dual.prefill == nullptr ||
        !dual.prefill->ok()) {
        throw std::runtime_error("internlm dual-profile engine did not create both contexts");
    }
    modules.prefill = std::move(dual.prefill);
    modules.decode = std::move(dual.decode);
    return modules;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    DecoderModules modules = load_modules(context, config);
    const cudaStream_t stream = modules.decode->stream();
    const std::int32_t kv_dim =
        (config.num_key_value_heads / config.tensor_parallel_size) * config.head_dim;
    auto state = std::make_unique<InternlmKvCache>(config.num_layers, config.max_cache_length,
                                                   kv_dim, stream, cache_dtype(config.precision),
                                                   make_kv_names(config.num_layers));
    if (!state->ok())
        throw std::runtime_error("internlm failed to create KV cache");

    InternlmTextGenConfig text_config;
    text_config.vocab_size = config.vocab_size;
    text_config.id_bos = config.bos_token_id;
    text_config.id_eos = config.eos_token_id;
    text_config.chat_template_format =
        internlm_detect_chat_template_format(chat_template(context.reader));
    text_config.prefill_max_length = config.max_cache_length;
    text_config.num_layers = config.num_layers;

    return new InternlmTextGenerationPipeline(
        std::move(modules.decode), std::move(state), std::move(text_config),
        create_tokenizer(context.reader), std::move(modules.prefill),
        std::move(modules.distributed_owner));
}

} // namespace trtmc::internlm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::internlm::create(context);
}
