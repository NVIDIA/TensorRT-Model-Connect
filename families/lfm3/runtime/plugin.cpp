/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/conv_state.h"
#include "families/lfm3/runtime/hybrid_state.h"
#include "families/lfm3/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::lfm3 {

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
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("lfm3 runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("lfm3 runtime.json has invalid '") + name + "'");
    }
}

std::vector<std::int32_t> require_eos_ids(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("lfm3 runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return {value.get<std::int32_t>()};
    if (value.is_array() && !value.empty())
        return value.get<std::vector<std::int32_t>>();
    throw std::runtime_error("lfm3 runtime.json has invalid 'eos_token_id'");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("lfm3 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("lfm3 runtime.json must be an object");
    if (json.size() != 25)
        throw std::runtime_error("lfm3 runtime.json has an unexpected field set");
    if (!require_value<bool>(json, "native_kv_cache") ||
        require_value<std::int32_t>(json, "native_kv_contract_version") != 1) {
        throw std::runtime_error("lfm3 runtime.json requires native KV contract version 1");
    }
    const std::int32_t intermediate_size = require_value<std::int32_t>(json, "intermediate_size");
    const float norm_eps = require_value<float>(json, "norm_eps");
    const auto rope = require_value<nlohmann::json>(json, "rope_parameters");
    const auto window = require_value<std::int32_t>(json, "sliding_window");
    (void)require_value<bool>(json, "use_sandwich_norm");
    if (!rope.is_object() || window <= 0)
        throw std::runtime_error("lfm3 runtime.json contains invalid attention metadata");
    for (const char* kind : {"full_attention", "sliding_attention"}) {
        const auto& item = rope.at(kind);
        const auto dimension = require_value<std::int32_t>(item, "rotary_dim");
        const auto theta = require_value<float>(item, "rope_theta");
        if (dimension < 2 || dimension % 2 || !std::isfinite(theta) || theta <= 0)
            throw std::runtime_error("lfm3 runtime.json contains invalid RoPE");
    }
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
    };
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 ||
        config.num_attention_layers <= 0 || config.num_conv_layers <= 0 ||
        config.num_attention_layers + config.num_conv_layers != config.num_layers ||
        config.conv_cache_length <= 0 || config.conv_state_dim != config.hidden_size ||
        config.vocab_size <= 0 || config.max_cache_length <= 0 || config.eos_token_ids.empty()) {
        throw std::runtime_error("lfm3 runtime.json contains invalid geometry");
    }
    if (intermediate_size <= 0 || !std::isfinite(norm_eps) || norm_eps <= 0.0F ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers) || layout != "single" ||
        (config.precision != "fp16" && config.precision != "bf16")) {
        throw std::runtime_error("lfm3 runtime.json contains invalid model contract");
    }
    std::int32_t conv_count = 0;
    for (const auto& kind : layer_types) {
        if (kind == "conv")
            ++conv_count;
        else if (kind != "full_attention" && kind != "sliding_attention")
            throw std::runtime_error("lfm3 runtime.json has unsupported layer type");
    }
    if (conv_count != config.num_conv_layers ||
        config.hidden_size != config.num_attention_heads * config.head_dim ||
        config.num_attention_heads % config.num_key_value_heads || config.bos_token_id < 0 ||
        config.bos_token_id >= config.vocab_size)
        throw std::runtime_error("lfm3 runtime.json has inconsistent geometry");
    for (auto id : config.eos_token_ids)
        if (id < 0 || id >= config.vocab_size)
            throw std::runtime_error("lfm3 runtime.json has invalid EOS id");
    return config;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    auto decoder = load_module(context.backend, require_section(context.reader, "engine.plan"));
    const cudaStream_t stream = decoder->stream();
    const DType dtype = state_dtype(config.precision);

    auto kv = std::make_unique<Lfm3KvCache>(config.num_attention_layers, config.max_cache_length,
                                            config.num_key_value_heads, config.head_dim, stream,
                                            dtype, kv_names(config.num_attention_layers));
    auto conv = std::make_unique<Lfm3ConvState>(config.num_conv_layers, config.conv_state_dim,
                                                config.conv_cache_length, stream, dtype);
    auto state = std::make_unique<Lfm3HybridState>(std::move(kv), std::move(conv));
    if (!state->ok())
        throw std::runtime_error("lfm3 failed to allocate inference state");

    Lfm3TextGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos = config.eos_token_ids.front();
    generation.id_eos_ids = config.eos_token_ids;
    apply_chat_template(context.reader, generation);

    return new Lfm3TextGenerationPipeline(std::move(decoder), std::move(state),
                                          std::move(generation), stream,
                                          create_tokenizer(context.reader), std::string{});
}

} // namespace trtmc::lfm3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lfm3 does not support --kv-cache-size");
    return trtmc::lfm3::create(context);
}
