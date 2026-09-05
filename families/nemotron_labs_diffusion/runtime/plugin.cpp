/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_labs_diffusion/runtime/chat_templates.h"
#include "families/nemotron_labs_diffusion/runtime/kv_cache.h"
#include "families/nemotron_labs_diffusion/runtime/pipeline.h"
#include "families/nemotron_labs_diffusion/runtime/plugin_helpers.h"
#include "families/nemotron_labs_diffusion/runtime/runtime_config.h"
#include "families/nemotron_labs_diffusion/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::nemotron_labs_diffusion {

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
    std::string precision;
    std::string lora_section;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("nemotron_labs_diffusion runtime.json missing '") +
                                 name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("nemotron_labs_diffusion runtime.json has invalid '") +
                                 name + "'");
    }
}

std::int32_t require_eos(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("nemotron_labs_diffusion runtime.json missing eos_token_id");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return value.get<std::int32_t>();
    if (value.is_array() && !value.empty() && value.front().is_number_integer())
        return value.front().get<std::int32_t>();
    throw std::runtime_error("nemotron_labs_diffusion runtime.json has invalid eos_token_id");
}

RuntimeConfig parse_runtime_config(std::string_view text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("nemotron_labs_diffusion invalid runtime.json: " +
                                 std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("nemotron_labs_diffusion runtime.json must be an object");
    if (json.size() != 11 && json.size() != 12)
        throw std::runtime_error(
            "nemotron_labs_diffusion runtime.json has an unexpected field set");
    if (json.size() == 12 && !json.contains("linear_spec_lora_engine_section"))
        throw std::runtime_error(
            "nemotron_labs_diffusion runtime.json has an unexpected optional field");
    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos(json),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::string>(json, "precision"),
        json.contains("linear_spec_lora_engine_section")
            ? require_value<std::string>(json, "linear_spec_lora_engine_section")
            : std::string{},
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.max_cache_length <= 0 ||
        config.num_key_value_heads * config.head_dim > config.hidden_size) {
        throw std::runtime_error("nemotron_labs_diffusion runtime.json contains invalid geometry");
    }
    if (config.precision != "fp16" && config.precision != "bf16" && config.precision != "fp32") {
        throw std::runtime_error("nemotron_labs_diffusion runtime.json has invalid precision");
    }
    return config;
}

DType cache_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

NemotronLabsDiffusionKvCacheNames make_kv_names(std::int32_t num_layers) {
    NemotronLabsDiffusionKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(nemotron_labs_diffusion_layer_tensor_name("cache_k", layer));
        names.cache_v.push_back(nemotron_labs_diffusion_layer_tensor_name("cache_v", layer));
        names.present_k.push_back(nemotron_labs_diffusion_layer_tensor_name("present_k", layer));
        names.present_v.push_back(nemotron_labs_diffusion_layer_tensor_name("present_v", layer));
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

std::unique_ptr<ITrtModule> load_lora_prefill(IBackend& backend, const BundleReader& bundle,
                                              const std::string& section_name) {
    if (section_name.empty())
        return nullptr;
    const auto& plan = require_section(bundle, section_name);
    auto dual = backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (dual.prefill == nullptr || !dual.prefill->ok())
        throw std::runtime_error("nemotron_labs_diffusion invalid LoRA prefill engine");
    return std::move(dual.prefill);
}

} // namespace

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(json);
}

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config =
        parse_runtime_config(require_text_section(context.reader, "runtime.json"));
    const auto& plan = require_section(context.reader, "engine.plan");
    auto dual = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (dual.prefill == nullptr || !dual.prefill->ok() || dual.decode == nullptr ||
        !dual.decode->ok()) {
        throw std::runtime_error("nemotron_labs_diffusion requires a dual-profile engine");
    }
    const cudaStream_t stream = dual.decode->stream();
    const std::int32_t kv_dim = config.num_key_value_heads * config.head_dim;
    auto state = std::make_unique<NemotronLabsDiffusionKvCache>(
        config.num_layers, config.max_cache_length, kv_dim, stream, cache_dtype(config.precision),
        make_kv_names(config.num_layers));
    if (!state->ok())
        throw std::runtime_error("nemotron_labs_diffusion failed to create KV cache");

    NemotronLabsDiffusionTextGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos = config.eos_token_id;
    generation.chat_template_format =
        nemotron_labs_diffusion_detect_chat_template_format(chat_template(context.reader));
    generation.prefill_max_length = config.max_cache_length;
    generation.num_layers = config.num_layers;
    generation.mask_token_id = 100;
    generation.diffusion_block_length = 32;
    generation.supports_text_diffusion = true;

    auto lora_prefill = load_lora_prefill(context.backend, context.reader, config.lora_section);
    return new NemotronLabsDiffusionTextGenerationPipeline(
        std::move(dual.decode), std::move(state), std::move(generation),
        create_tokenizer(context.reader), std::move(dual.prefill), std::move(lora_prefill),
        nullptr);
}

} // namespace trtmc::nemotron_labs_diffusion

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_labs_diffusion does not support --kv-cache-size");
    return trtmc::nemotron_labs_diffusion::create(context);
}
