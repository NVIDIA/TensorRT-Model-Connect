/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mamba/runtime/chat_templates.h"
#include "families/mamba/runtime/distributed_runtime.h"
#include "families/mamba/runtime/pipeline.h"
#include "families/mamba/runtime/plugin_helpers.h"
#include "families/mamba/runtime/recurrent_state.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::mamba {

namespace {

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("mamba runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("mamba runtime.json has invalid '") + name + "'");
    }
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
    nlohmann::json json;
    try {
        const std::string text = require_text_section(context.reader, "runtime.json");
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("mamba invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("mamba runtime.json must be an object");
    if (json.size() != 17)
        throw std::runtime_error("mamba runtime.json has an unexpected field set");

    const std::int32_t hidden_size = require_value<std::int32_t>(json, "hidden_size");
    const std::int32_t num_layers = require_value<std::int32_t>(json, "num_hidden_layers");
    const std::int32_t vocab_size = require_value<std::int32_t>(json, "vocab_size");
    const std::int32_t bos = require_value<std::int32_t>(json, "bos_token_id");
    const std::int32_t eos = require_value<std::int32_t>(json, "eos_token_id");
    const std::int32_t num_attention_heads =
        require_value<std::int32_t>(json, "num_attention_heads");
    const std::int32_t num_key_value_heads =
        require_value<std::int32_t>(json, "num_key_value_heads");
    const std::int32_t head_dim = require_value<std::int32_t>(json, "head_dim");
    (void)require_value<std::int32_t>(json, "pad_token_id");
    const std::int32_t expand = require_value<std::int32_t>(json, "expand");
    const std::int32_t inner = hidden_size * expand;
    const std::int32_t state_size = require_value<std::int32_t>(json, "state_size");
    const std::int32_t conv_kernel = require_value<std::int32_t>(json, "conv_kernel");
    const std::int32_t tp_size = require_value<std::int32_t>(json, "tensor_parallel_size");
    const std::int32_t max_cache_length = require_value<std::int32_t>(json, "max_cache_length");
    const std::string precision = require_value<std::string>(json, "precision");
    const std::string tp_mode = require_value<std::string>(json, "tensor_parallel_mode");
    const std::string layout = require_value<std::string>(json, "decoder_engine_layout");
    if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0 || inner <= 0 || state_size <= 0 ||
        conv_kernel <= 1 || tp_size <= 0 || max_cache_length <= 0 || num_attention_heads <= 0 ||
        num_key_value_heads <= 0 || head_dim <= 0 || expand <= 0 ||
        (layout != "single" && layout != "dual_profile") ||
        tp_mode != (tp_size > 1 ? "tensor_parallel" : "single") ||
        (precision != "fp16" && precision != "bf16" && precision != "fp32")) {
        throw std::runtime_error("mamba runtime.json contains invalid geometry");
    }

    DistributedRuntimeGroup group = initialize_tensor_parallel_group(tp_size);
    ModuleCreateOptions options;
    if (tp_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string section =
        tp_size > 1 ? "engine.rank" + std::to_string(group.rank) + ".plan" : "engine.plan";
    const auto& plan = require_section(context.reader, section);
    auto decoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("mamba failed to load decoder");
    decoder->set_timing_label("mamba decoder");
    const cudaStream_t stream = decoder->stream();

    std::vector<MambaRecurrentState::TensorSpec> specs = {
        {"conv_state", {inner, conv_kernel}, "present_conv"},
        {"ssm_state", {inner, state_size}, "present_ssm"},
    };
    auto state = std::make_unique<MambaRecurrentState>(num_layers, std::move(specs), stream);
    if (!state->ok())
        throw std::runtime_error("mamba failed to create recurrent state");

    RecurrentGenConfig generation;
    generation.vocab_size = vocab_size;
    generation.id_bos = bos;
    generation.id_eos = eos;
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        mamba_detect_chat_template_format(chat_template(context.reader));
    return new RecurrentPipeline(std::move(decoder), std::move(state), std::move(generation),
                                 stream, "Mamba", create_tokenizer(context.reader), std::string{});
}

} // namespace trtmc::mamba

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::mamba::create(context);
}
