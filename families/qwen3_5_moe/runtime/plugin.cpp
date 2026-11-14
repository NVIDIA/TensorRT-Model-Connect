/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5_moe/runtime/chat_templates.h"
#include "families/qwen3_5_moe/runtime/hybrid_state.h"
#include "families/qwen3_5_moe/runtime/native_media.h"
#include "families/qwen3_5_moe/runtime/pipeline.h"
#include "families/qwen3_5_moe/runtime/plugin_helpers.h"
#include "families/qwen3_5_moe/runtime/recurrent_state.h"
#include "trtmc/runtime/family_factory.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/qwen3_5_moe/runtime/edge_llm/adapter.h"
#endif

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_5_moe {

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
    std::string kv_cache_quantization;
    bool has_mtp;
    std::int32_t mtp_num_draft_tokens;
    std::int32_t tensor_parallel_size;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("qwen3_5_moe runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("qwen3_5_moe runtime.json has invalid '") + name +
                                 "'");
    }
}

std::vector<std::int32_t> require_eos_ids(const nlohmann::json& json) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("qwen3_5_moe runtime.json missing eos_token_id");
    const auto& value = json.at("eos_token_id");
    if (value.is_number_integer())
        return {require_value<std::int32_t>(json, "eos_token_id")};
    if (!value.is_array() || value.empty() ||
        std::any_of(value.begin(), value.end(),
                    [](const auto& id) { return !id.is_number_integer(); }))
        throw std::runtime_error(
            "qwen3_5_moe eos_token_id must be an integer or nonempty integer array");
    return require_value<std::vector<std::int32_t>>(json, "eos_token_id");
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("qwen3_5_moe invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("qwen3_5_moe runtime.json must be an object");
    if (json.size() != 20 + static_cast<std::size_t>(json.contains("kv_cache_quantization")) +
                           (json.contains("has_mtp") ? 2 : 0) +
                           static_cast<std::size_t>(json.contains("tensor_parallel_size")))
        throw std::runtime_error("qwen3_5_moe runtime.json has an unexpected field set");
    if (require_value<std::string>(json, "decoder_engine_layout") != "single")
        throw std::runtime_error("qwen3_5_moe requires decoder_engine_layout='single'");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_eos_ids(json),
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
        json.contains("kv_cache_quantization")
            ? require_value<std::string>(json, "kv_cache_quantization")
            : "none",
        json.contains("has_mtp") ? require_value<bool>(json, "has_mtp") : false,
        json.contains("has_mtp") ? require_value<std::int32_t>(json, "mtp_num_draft_tokens") : 0,
        json.contains("tensor_parallel_size")
            ? require_value<std::int32_t>(json, "tensor_parallel_size")
            : 1,
    };
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.num_attention_layers <= 0 || config.num_mamba_layers <= 0 ||
        config.num_attention_layers + config.num_mamba_layers != config.num_layers ||
        config.d_inner <= 0 || config.mamba_d_state <= 0 || config.mamba_d_conv <= 0 ||
        config.mamba_nheads <= 0 || config.mamba_head_dim <= 0 || config.conv_dim <= 0 ||
        config.max_cache_length <= 0 ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers)) {
        throw std::runtime_error("qwen3_5_moe runtime.json contains invalid geometry");
    }
    if (config.precision != "fp16" && config.precision != "fp32")
        throw std::runtime_error("qwen3_5_moe runtime.json contains invalid precision");
    if (config.kv_cache_quantization != "none" && config.kv_cache_quantization != "fp8")
        throw std::runtime_error("qwen3_5_moe runtime.json contains invalid KV cache quantization");
    if ((config.has_mtp && config.mtp_num_draft_tokens != 3) ||
        (!config.has_mtp && config.mtp_num_draft_tokens != 0))
        throw std::runtime_error(
            "Qwen3.5-MoE native MTP requires three draft steps and four verification positions");
    if (config.tensor_parallel_size != 1)
        throw std::runtime_error("Qwen3.5-MoE native runtime requires TP1");
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

ITask* create_native(const FamilyContext& context, const RuntimeConfig& config) {
    const auto plan = require_section(context.reader, "engine.plan");
    auto decoder = context.backend.create_module(plan.data(), plan.size(), {});
    if (!decoder || !decoder->ok())
        throw std::runtime_error("Qwen3.5-MoE native decoder failed to load");
    decoder->set_timing_label("qwen3_5_moe decoder");
    const cudaStream_t stream = decoder->stream();
    const std::int32_t kv_dim =
        config.num_key_value_heads / config.tensor_parallel_size * config.head_dim;

    const auto cache_dtype =
        config.kv_cache_quantization == "fp8" ? DType::kFloat8E4M3 : state_dtype(config.precision);
    auto kv = std::make_unique<Qwen35MoeKvCache>(
        config.num_attention_layers, config.max_cache_length, kv_dim, stream, cache_dtype);
    if (!kv->ok())
        throw std::runtime_error("qwen3_5_moe failed to create KV cache");

    const std::int64_t conv_elements =
        static_cast<std::int64_t>(config.conv_dim / config.tensor_parallel_size) *
        config.mamba_d_conv;
    const std::int64_t state_elements =
        static_cast<std::int64_t>(config.mamba_nheads / config.tensor_parallel_size) *
        config.mamba_head_dim * config.mamba_d_state;
    auto recurrent = std::make_unique<Qwen35MoeRecurrentState>(
        config.num_mamba_layers,
        std::vector<Qwen35MoeRecurrentState::TensorSpec>{
            {"conv_state", {conv_elements}, "present_conv"},
            {"ssm_state", {state_elements}, "present_ssm"},
        },
        stream);
    auto state = std::make_unique<Qwen35MoeHybridState>(std::move(kv), std::move(recurrent));
    if (!state->ok())
        throw std::runtime_error("qwen3_5_moe failed to create hybrid state");

    std::unique_ptr<Qwen35MoeMtpScheduler> mtp_scheduler;
    if (config.has_mtp) {
        auto draft =
            load_engine(context.backend, require_section(context.reader, "mtp_engine.plan"),
                        "qwen3_5_moe MTP draft");
        auto verifier =
            load_engine(context.backend, require_section(context.reader, "multi_token_engine.plan"),
                        "qwen3_5_moe MTP verifier");
        mtp_scheduler = std::make_unique<Qwen35MoeMtpScheduler>(
            std::move(draft), nullptr, std::move(verifier), *state, config.hidden_size,
            config.vocab_size, config.mtp_num_draft_tokens, stream);
    } else if (context.reader.find_section("mtp_engine.plan") ||
               context.reader.find_section("multi_token_engine.plan")) {
        throw std::runtime_error(
            "Qwen3.5-MoE MTP engine sections require explicit runtime metadata");
    }

    auto paired_draft = create_qwen35_paired(
        context, config.hidden_size, config.num_layers, config.vocab_size, config.max_cache_length,
        config.precision, config.tensor_parallel_size, config.has_mtp);

    RecurrentGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.id_eos_ids = config.eos_token_ids;
    generation.id_eos = config.eos_token_ids.front();
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        qwen3_5_moe_detect_chat_template_format(chat_template(context.reader));

    auto tokenizer = create_tokenizer(context.reader);
    auto pipeline = std::make_unique<RecurrentPipeline>(
        std::move(decoder), std::move(state), std::move(generation), stream, "Qwen3.5-MoE",
        tokenizer, std::string{}, std::move(mtp_scheduler), std::move(paired_draft));
    if (context.reader.find_section("native_visual.json")) {
        auto visual =
            load_engine(context.backend, require_section(context.reader, "native_visual.plan"),
                        "qwen3_5_moe native visual");
        return create_native_media_task(context.reader, std::move(pipeline), std::move(visual),
                                        std::move(tokenizer));
    }
    if (context.reader.info().task != "text_generation")
        throw std::runtime_error("Qwen3.5-MoE native media bundle is missing visual metadata");
    return pipeline.release();
}

ITask* create(const FamilyContext& context) {
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_HAS_EDGE_LLM
        return edge_llm::create(context.reader);
#else
        throw std::runtime_error("Qwen3.5-MoE Edge bundle requires a runtime configured with "
                                 "-DTRTMC_ENABLE_EDGELLM=ON; rebuild and install Model Connect");
#endif
    }
    const auto config = parse_runtime_config(context.reader);
    if (config.tensor_parallel_size != 1)
        throw std::runtime_error("Qwen3.5-MoE native runtime requires TP1");
    return create_native(context, config);
}

} // namespace trtmc::qwen3_5_moe

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_5_moe does not support --kv-cache-size");
    return trtmc::qwen3_5_moe::create(context);
}
