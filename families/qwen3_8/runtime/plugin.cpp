/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/chat_templates.h"
#include "families/qwen3_8/runtime/hybrid_state.h"
#include "families/qwen3_8/runtime/pipeline.h"
#include "families/qwen3_8/runtime/plugin_helpers.h"
#include "families/qwen3_8/runtime/recurrent_state.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_8 {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_attention_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t num_attention_layers;
    std::int32_t num_mamba_layers;
    std::int32_t d_inner;
    std::int32_t mamba_d_state;
    std::int32_t mamba_d_conv;
    std::int32_t mamba_nheads;
    std::int32_t mamba_head_dim;
    std::int32_t conv_dim;
    std::int32_t max_cache_length;
    std::string precision;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("qwen3_8 runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("qwen3_8 runtime.json has invalid '") + name + "'");
    }
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("qwen3_8 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("qwen3_8 runtime.json must be an object");
    if (json.size() != 20)
        throw std::runtime_error("qwen3_8 runtime.json has an unexpected field set");
    if (require_value<std::string>(json, "decoder_engine_layout") != "single")
        throw std::runtime_error("qwen3_8 requires decoder_engine_layout='single'");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_value<std::vector<std::int32_t>>(json, "eos_token_ids"),
        require_value<std::int32_t>(json, "num_attention_layers"),
        require_value<std::int32_t>(json, "num_mamba_layers"),
        require_value<std::int32_t>(json, "d_inner"),
        require_value<std::int32_t>(json, "mamba_d_state"),
        require_value<std::int32_t>(json, "mamba_d_conv"),
        require_value<std::int32_t>(json, "mamba_nheads"),
        require_value<std::int32_t>(json, "mamba_head_dim"),
        require_value<std::int32_t>(json, "conv_dim"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::string>(json, "precision"),
    };
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.num_attention_layers <= 0 || config.num_mamba_layers <= 0 ||
        config.num_attention_layers + config.num_mamba_layers != config.num_layers ||
        config.d_inner <= 0 || config.mamba_d_state <= 0 || config.mamba_d_conv <= 0 ||
        config.mamba_nheads <= 0 || config.mamba_head_dim <= 0 || config.conv_dim <= 0 ||
        config.max_cache_length <= 0 || config.eos_token_ids.empty() ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers)) {
        throw std::runtime_error("qwen3_8 runtime.json contains invalid geometry");
    }
    if (config.precision != "fp16" && config.precision != "fp32")
        throw std::runtime_error("qwen3_8 runtime.json contains invalid precision");
    return config;
}

DType state_dtype(const std::string& precision) {
    return precision == "fp16" ? DType::kFloat16 : DType::kFloat32;
}

std::string chat_template(const BundleReader& bundle) {
    const auto* section = bundle.find_section("chat_template.jinja");
    if (section == nullptr || section->length == 0)
        return {};
    const auto data = bundle.read_section("chat_template.jinja");
    return {data.begin(), data.end()};
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    auto decoder = load_engine(context.backend, require_section(context.reader, "engine.plan"),
                               "qwen3_8 decoder");
    const cudaStream_t stream = decoder->stream();
    const std::int32_t kv_dim = config.num_key_value_heads * config.head_dim;

    auto kv = std::make_unique<Qwen38KvCache>(config.num_attention_layers, config.max_cache_length,
                                              kv_dim, stream, state_dtype(config.precision));
    if (!kv->ok())
        throw std::runtime_error("qwen3_8 failed to create KV cache");

    const std::int64_t conv_elements =
        static_cast<std::int64_t>(config.conv_dim) * config.mamba_d_conv;
    const std::int64_t state_elements = static_cast<std::int64_t>(config.mamba_nheads) *
                                        config.mamba_head_dim * config.mamba_d_state;
    auto recurrent =
        std::make_unique<Qwen38RecurrentState>(config.num_mamba_layers,
                                               std::vector<Qwen38RecurrentState::TensorSpec>{
                                                   {"conv_state", {conv_elements}, "present_conv"},
                                                   {"ssm_state", {state_elements}, "present_ssm"},
                                               },
                                               stream);
    auto state = std::make_unique<Qwen38HybridState>(std::move(kv), std::move(recurrent));
    if (!state->ok())
        throw std::runtime_error("qwen3_8 failed to create hybrid state");

    RecurrentGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos_ids = config.eos_token_ids;
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        qwen3_8_detect_chat_template_format(chat_template(context.reader));

    return new RecurrentPipeline(std::move(decoder), std::move(state), std::move(generation),
                                 stream, "Qwen3.8", create_tokenizer(context.reader),
                                 std::string{});
}

} // namespace trtmc::qwen3_8

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_8 does not support --kv-cache-size");
    return trtmc::qwen3_8::create(context);
}
