/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/conv_state.h"
#include "families/lfm2/runtime/hybrid_state.h"
#include "families/lfm2/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::lfm2 {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_attention_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t num_attention_layers;
    std::int32_t num_conv_layers;
    std::int32_t conv_cache_length;
    std::int32_t conv_state_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t max_cache_length;
    std::string precision;
    // "single": one single-token profile (prompt fed one token per launch).
    // "dual_profile": profile 0 prefills a chunk of tokens, profile 1 decodes one.
    bool dual_profile;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("lfm2 runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("lfm2 runtime.json has invalid '") + name + "'");
    }
}

std::vector<std::int32_t> require_eos_ids(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("lfm2 runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return {value.get<std::int32_t>()};
    if (value.is_array() && !value.empty())
        return value.get<std::vector<std::int32_t>>();
    throw std::runtime_error("lfm2 runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("lfm2 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("lfm2 runtime.json must be an object");
    if (json.size() != 23)
        throw std::runtime_error("lfm2 runtime.json has an unexpected field set");
    if (!require_value<bool>(json, "native_kv_cache") ||
        require_value<std::int32_t>(json, "native_kv_contract_version") != 1) {
        throw std::runtime_error("lfm2 runtime.json requires native KV contract version 1");
    }
    const std::int32_t intermediate_size = require_value<std::int32_t>(json, "intermediate_size");
    const float norm_eps = require_value<float>(json, "norm_eps");
    const float rope_theta = require_value<float>(json, "rope_theta");
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    (void)require_value<bool>(json, "tie_word_embeddings");
    (void)require_value<std::int32_t>(json, "pad_token_id");
    const std::string layout = require_value<std::string>(json, "decoder_engine_layout");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "num_attention_layers"),
        require_value<std::int32_t>(json, "num_conv_layers"),
        require_value<std::int32_t>(json, "conv_L_cache"),
        require_value<std::int32_t>(json, "conv_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_ids(json),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::string>(json, "precision"),
        layout == "dual_profile",
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 ||
        config.num_attention_layers <= 0 || config.num_conv_layers <= 0 ||
        config.num_attention_layers + config.num_conv_layers != config.num_layers ||
        config.conv_cache_length <= 0 || config.conv_state_dim != config.hidden_size ||
        config.vocab_size <= 0 || config.max_cache_length <= 0 || config.eos_token_ids.empty()) {
        throw std::runtime_error("lfm2 runtime.json contains invalid geometry");
    }
    if (intermediate_size <= 0 || norm_eps <= 0.0F || rope_theta <= 0.0F ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers) ||
        (layout != "single" && layout != "dual_profile") ||
        (config.precision != "fp16" && config.precision != "bf16")) {
        throw std::runtime_error("lfm2 runtime.json contains invalid model contract");
    }
    return config;
}

// A dual-profile context accepts token inputs of [1..max_tokens].
void require_token_profile(const ITrtModule& module, const std::string& name,
                           std::int64_t max_tokens) {
    const auto profile = module.profile_idx();
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.input_profile_shape(name, profile, ProfileShapeSelector::kMin) !=
            std::vector<std::int64_t>{1} ||
        module.input_profile_shape(name, profile, ProfileShapeSelector::kMax) !=
            std::vector<std::int64_t>{max_tokens}) {
        throw std::runtime_error("lfm2 engine input profile mismatch for '" + name + "'");
    }
}

struct DecoderModules {
    std::unique_ptr<ITrtModule> decode;
    std::unique_ptr<ITrtModule> prefill;
};

DecoderModules load_decoder(const FamilyContext& context, const RuntimeConfig& config) {
    const auto plan = require_section(context.reader, "engine.plan");
    if (!config.dual_profile) {
        auto decoder = load_module(context.backend, plan);
        if (decoder->optimization_profile_count() != 1)
            throw std::runtime_error("lfm2 single-layout engine needs exactly one profile");
        return {std::move(decoder), nullptr};
    }
    auto modules = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (modules.prefill == nullptr || !modules.prefill->ok() || modules.decode == nullptr ||
        !modules.decode->ok()) {
        throw std::runtime_error("lfm2 dual-profile engine did not create both contexts");
    }
    if (modules.decode->optimization_profile_count() != 2)
        throw std::runtime_error("lfm2 dual-profile engine needs two profiles");
    const auto chunk = modules.prefill->input_profile_shape(
        "token_id", modules.prefill->profile_idx(), ProfileShapeSelector::kMax);
    if (chunk.size() != 1 || chunk[0] < 1 || chunk[0] > config.max_cache_length)
        throw std::runtime_error("lfm2 prefill profile has an invalid token range");
    for (const char* name : {"token_id", "position_id"}) {
        require_token_profile(*modules.prefill, name, chunk[0]);
        require_token_profile(*modules.decode, name, 1);
    }
    modules.prefill->set_timing_label("lfm2 prefill engine");
    modules.decode->set_timing_label("lfm2 engine");
    return {std::move(modules.decode), std::move(modules.prefill)};
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    auto modules = load_decoder(context, config);
    auto& decoder = modules.decode;
    const cudaStream_t stream = decoder->stream();
    const DType dtype = state_dtype(config.precision);

    auto kv = std::make_unique<Lfm2KvCache>(config.num_attention_layers, config.max_cache_length,
                                            config.num_key_value_heads, config.head_dim, stream,
                                            dtype, kv_names(config.num_attention_layers));
    auto conv = std::make_unique<Lfm2ConvState>(config.num_conv_layers, config.conv_state_dim,
                                                config.conv_cache_length, stream, dtype);
    auto state = std::make_unique<Lfm2HybridState>(std::move(kv), std::move(conv));
    if (!state->ok())
        throw std::runtime_error("lfm2 failed to allocate inference state");

    Lfm2TextGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos = config.eos_token_ids.front();
    generation.id_eos_ids = config.eos_token_ids;
    apply_chat_template(context.reader, generation);

    return new Lfm2TextGenerationPipeline(
        std::move(decoder), std::move(state), std::move(generation), stream,
        create_tokenizer(context.reader), std::string{}, std::move(modules.prefill));
}

} // namespace trtmc::lfm2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lfm2 does not support --kv-cache-size");
    return trtmc::lfm2::create(context);
}
