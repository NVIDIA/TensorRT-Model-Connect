/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_h/runtime/chat_templates.h"
#include "families/nemotron_h/runtime/distributed_runtime.h"
#include "families/nemotron_h/runtime/hybrid_state.h"
#include "families/nemotron_h/runtime/pipeline.h"
#include "families/nemotron_h/runtime/plugin_helpers.h"
#include "families/nemotron_h/runtime/recurrent_state.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::nemotron_h {

namespace {

struct RuntimeConfig {
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_attention_heads;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t vocab_size;
    std::int32_t bos_token_id;
    std::int32_t eos_token_id;
    std::vector<std::int32_t> stop_token_ids;
    std::int32_t pad_token_id;
    std::int32_t num_attention_layers;
    std::int32_t num_mamba_layers;
    std::int32_t d_inner;
    std::int32_t mamba_d_state;
    std::int32_t mamba_d_conv;
    std::int32_t mamba_nheads;
    std::int32_t mamba_head_dim;
    std::int32_t conv_dim;
    std::int32_t max_cache_length;
    std::int32_t tensor_parallel_size;
    std::string tensor_parallel_mode;
    std::string precision;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("nemotron_h runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("nemotron_h runtime.json has invalid '") + name + "'");
    }
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const std::string text = require_text_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("nemotron_h invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("nemotron_h runtime.json must be an object");
    if (json.size() != 25)
        throw std::runtime_error("nemotron_h runtime.json has an unexpected field set");
    const std::string layout = require_value<std::string>(json, "decoder_engine_layout");
    if (layout != "single" && layout != "dual_profile")
        throw std::runtime_error("nemotron_h runtime.json has invalid decoder_engine_layout");

    RuntimeConfig config{
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_attention_heads"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_value<std::int32_t>(json, "eos_token_id"),
        require_value<std::vector<std::int32_t>>(json, "stop_token_ids"),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "num_attention_layers"),
        require_value<std::int32_t>(json, "num_mamba_layers"),
        require_value<std::int32_t>(json, "d_inner"),
        require_value<std::int32_t>(json, "mamba_d_state"),
        require_value<std::int32_t>(json, "mamba_d_conv"),
        require_value<std::int32_t>(json, "mamba_nheads"),
        require_value<std::int32_t>(json, "mamba_head_dim"),
        require_value<std::int32_t>(json, "conv_dim"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::int32_t>(json, "tensor_parallel_size"),
        require_value<std::string>(json, "tensor_parallel_mode"),
        require_value<std::string>(json, "precision"),
    };
    const auto layer_types = require_value<std::vector<std::string>>(json, "layer_types");
    const std::int32_t groups = require_value<std::int32_t>(json, "n_groups");
    if (config.hidden_size <= 0 || config.num_layers <= 0 || config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.num_attention_layers <= 0 || config.num_mamba_layers <= 0 ||
        config.num_attention_layers + config.num_mamba_layers != config.num_layers ||
        config.d_inner <= 0 || config.mamba_d_state <= 0 || config.mamba_d_conv <= 0 ||
        config.mamba_nheads <= 0 || config.mamba_head_dim <= 0 || config.conv_dim <= 0 ||
        config.max_cache_length <= 0 || config.tensor_parallel_size <= 0 || groups <= 0 ||
        layer_types.size() != static_cast<std::size_t>(config.num_layers) ||
        config.tensor_parallel_mode !=
            (config.tensor_parallel_size > 1 ? "tensor_parallel" : "single")) {
        throw std::runtime_error("nemotron_h runtime.json contains invalid geometry");
    }
    if (config.stop_token_ids.empty() || config.stop_token_ids.front() != config.eos_token_id)
        throw std::runtime_error("nemotron_h runtime.json contains invalid stop tokens");
    std::vector<std::int32_t> unique_stop_tokens = config.stop_token_ids;
    std::sort(unique_stop_tokens.begin(), unique_stop_tokens.end());
    if (unique_stop_tokens.front() < 0 || unique_stop_tokens.back() >= config.vocab_size ||
        std::adjacent_find(unique_stop_tokens.begin(), unique_stop_tokens.end()) !=
            unique_stop_tokens.end()) {
        throw std::runtime_error("nemotron_h runtime.json contains invalid stop tokens");
    }
    if (config.precision != "fp16" && config.precision != "bf16" && config.precision != "fp32")
        throw std::runtime_error("nemotron_h runtime.json contains invalid precision");
    if (config.num_key_value_heads % config.tensor_parallel_size != 0)
        throw std::runtime_error("nemotron_h KV heads must be divisible by tensor parallel size");
    return config;
}

DType state_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
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
    DistributedRuntimeGroup group = initialize_tensor_parallel_group(config.tensor_parallel_size);
    ModuleCreateOptions options;
    if (config.tensor_parallel_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string section = config.tensor_parallel_size > 1
                                    ? "engine.rank" + std::to_string(group.rank) + ".plan"
                                    : std::string("engine.plan");
    const auto& plan = require_section(context.reader, section);
    auto decoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("nemotron_h failed to load decoder");
    decoder->set_timing_label("nemotron_h decoder");
    const cudaStream_t stream = decoder->stream();
    const std::int32_t kv_dim =
        (config.num_key_value_heads / config.tensor_parallel_size) * config.head_dim;

    auto kv =
        std::make_unique<NemotronHKvCache>(config.num_attention_layers, config.max_cache_length,
                                           kv_dim, stream, state_dtype(config.precision));
    if (!kv->ok())
        throw std::runtime_error("nemotron_h failed to create KV cache");

    const std::int64_t conv_elements =
        static_cast<std::int64_t>(config.conv_dim) * config.mamba_d_conv;
    const std::int64_t state_elements = static_cast<std::int64_t>(config.mamba_nheads) *
                                        config.mamba_head_dim * config.mamba_d_state;
    auto recurrent = std::make_unique<NemotronHRecurrentState>(
        config.num_mamba_layers,
        std::vector<NemotronHRecurrentState::TensorSpec>{
            {"conv_state", {conv_elements}, "present_conv"},
            {"ssm_state", {state_elements}, "present_ssm"},
        },
        stream);
    auto state = std::make_unique<NemotronHHybridState>(std::move(kv), std::move(recurrent));
    if (!state->ok())
        throw std::runtime_error("nemotron_h failed to create hybrid state");

    RecurrentGenConfig generation;
    generation.vocab_size = config.vocab_size;
    generation.id_bos = config.bos_token_id;
    generation.stop_token_ids = config.stop_token_ids;
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        nemotron_h_detect_chat_template_format(chat_template(context.reader));

    return new RecurrentPipeline(std::move(decoder), std::move(state), std::move(generation),
                                 stream, "Nemotron-H", create_tokenizer(context.reader),
                                 std::string{});
}

} // namespace trtmc::nemotron_h

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::nemotron_h::create(context);
}
