/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rwkv/runtime/chat_templates.h"
#include "families/rwkv/runtime/distributed_runtime.h"
#include "families/rwkv/runtime/pipeline.h"
#include "families/rwkv/runtime/plugin_helpers.h"
#include "families/rwkv/runtime/recurrent_state.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::rwkv {

namespace {

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("rwkv runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("rwkv runtime.json has invalid '") + name + "'");
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
        json = nlohmann::json::parse(require_text_section(context.reader, "runtime.json"));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("rwkv invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("rwkv runtime.json must be an object");
    if (json.size() != 14)
        throw std::runtime_error("rwkv runtime.json has an unexpected field set");

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
    const std::int32_t tp_size = require_value<std::int32_t>(json, "tensor_parallel_size");
    const std::int32_t max_cache_length = require_value<std::int32_t>(json, "max_cache_length");
    const std::string precision = require_value<std::string>(json, "precision");
    const std::string tp_mode = require_value<std::string>(json, "tensor_parallel_mode");
    const std::string layout = require_value<std::string>(json, "decoder_engine_layout");
    if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0 || tp_size <= 0 ||
        max_cache_length <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 ||
        head_dim <= 0 || (layout != "single" && layout != "dual_profile") ||
        tp_mode != (tp_size > 1 ? "tensor_parallel" : "single") ||
        (precision != "fp16" && precision != "bf16" && precision != "fp32")) {
        throw std::runtime_error("rwkv runtime.json contains invalid geometry");
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
        throw std::runtime_error("rwkv failed to load decoder");
    decoder->set_timing_label("rwkv decoder");
    const cudaStream_t stream = decoder->stream();

    std::vector<RwkvRecurrentState::TensorSpec> specs = {
        {"attn_state", {hidden_size}, "present_attn"}, {"ff_state", {hidden_size}, "present_ff"},
        {"num_state", {hidden_size}, "present_num"},   {"den_state", {hidden_size}, "present_den"},
        {"max_state", {hidden_size}, "present_max"},
    };
    auto state = std::make_unique<RwkvRecurrentState>(num_layers, std::move(specs), stream);
    if (!state->ok())
        throw std::runtime_error("rwkv failed to create recurrent state");

    RecurrentGenConfig generation;
    generation.vocab_size = vocab_size;
    generation.id_bos = bos;
    generation.id_eos = eos;
    generation.has_position_input = decoder->has_input("position_id");
    generation.chat_template_format =
        rwkv_detect_chat_template_format(chat_template(context.reader));
    return new RecurrentPipeline(std::move(decoder), std::move(state), std::move(generation),
                                 stream, "RWKV", create_tokenizer(context.reader), std::string{});
}

} // namespace trtmc::rwkv

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("rwkv does not support --kv-cache-size");
    return trtmc::rwkv::create(context);
}
