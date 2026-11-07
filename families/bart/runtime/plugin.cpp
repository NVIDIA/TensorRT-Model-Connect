/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BartPlugin: handles "bart_seq2seq_encoder_decoder" strategy.
// Encoder-decoder text-to-text pipeline for BART models.

#include "families/bart/runtime/decode_runtime.h"
#include "families/bart/runtime/device_buffer.h"
#include "families/bart/runtime/distributed_runtime.h"
#include "families/bart/runtime/kv_cache.h"
#include "families/bart/runtime/pipeline.h"
#include "families/bart/runtime/plugin_helpers.h"
#include "families/bart/runtime/runtime_config.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct BartRuntimeConfig {
    int32_t hidden_size;
    int32_t decoder_layers;
    int32_t decoder_attention_heads;
    int32_t max_source_length;
    int32_t vocab_size;
    int32_t decoder_start_token_id;
    int32_t eos_token_id;
    int32_t bos_token_id;
    int32_t pad_token_id;
    int32_t max_cache_length;
    int32_t tensor_parallel_size;
    std::string precision;
    std::string tensor_parallel_mode;
    std::string decoder_engine_layout;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("bart runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("bart runtime.json has invalid '") + name + "'");
    }
}

BartRuntimeConfig parse_runtime_config(std::string_view text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("bart invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("bart runtime.json must be an object");
    if (json.size() != 21)
        throw std::runtime_error("bart runtime.json has an unexpected field set");
    const int32_t encoder_layers = require_value<int32_t>(json, "encoder_layers");
    const int32_t encoder_heads = require_value<int32_t>(json, "encoder_attention_heads");
    const int32_t encoder_ffn = require_value<int32_t>(json, "encoder_ffn_dim");
    const int32_t decoder_ffn = require_value<int32_t>(json, "decoder_ffn_dim");
    const int32_t max_positions = require_value<int32_t>(json, "max_position_embeddings");
    const int32_t position_offset = require_value<int32_t>(json, "position_embedding_offset");
    const bool has_encoder = require_value<bool>(json, "has_vision_engine");
    const bool is_encoder_decoder = require_value<bool>(json, "is_encoder_decoder");
    BartRuntimeConfig config{
        require_value<int32_t>(json, "hidden_size"),
        require_value<int32_t>(json, "decoder_layers"),
        require_value<int32_t>(json, "decoder_attention_heads"),
        require_value<int32_t>(json, "max_cache_length"),
        require_value<int32_t>(json, "vocab_size"),
        require_value<int32_t>(json, "decoder_start_token_id"),
        require_value<int32_t>(json, "eos_token_id"),
        require_value<int32_t>(json, "bos_token_id"),
        require_value<int32_t>(json, "pad_token_id"),
        require_value<int32_t>(json, "max_cache_length"),
        require_value<int32_t>(json, "tensor_parallel_size"),
        require_value<std::string>(json, "precision"),
        require_value<std::string>(json, "tensor_parallel_mode"),
        require_value<std::string>(json, "decoder_engine_layout"),
    };
    if (config.hidden_size <= 0 || config.decoder_layers <= 0 ||
        config.decoder_attention_heads <= 0 ||
        config.hidden_size % config.decoder_attention_heads != 0 || config.max_source_length <= 0 ||
        config.vocab_size <= 0 || config.max_cache_length <= 0 ||
        config.tensor_parallel_size <= 0 || encoder_layers <= 0 || encoder_heads <= 0 ||
        encoder_ffn <= 0 || decoder_ffn <= 0 || max_positions <= 0 || !has_encoder ||
        !is_encoder_decoder || position_offset != 2 ||
        config.tensor_parallel_mode !=
            (config.tensor_parallel_size > 1 ? "tensor_parallel" : "single") ||
        config.decoder_engine_layout !=
            (config.tensor_parallel_size > 1 ? "dual_profile" : "single") ||
        (config.precision != "fp16" && config.precision != "fp32")) {
        throw std::runtime_error("bart runtime.json contains invalid geometry");
    }
    return config;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine.rank" + std::to_string(rank) + ".plan";
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const ITrtModule& module, const BartRuntimeConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : config.hidden_size;
}

} // namespace

namespace bart {

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(json);
}

} // namespace bart

ITask* create_bart(const FamilyContext& context) {
    const BartRuntimeConfig config =
        parse_runtime_config(bart::require_text_section(context.reader, "runtime.json"));
    bart::DistributedRuntimeGroup group =
        bart::initialize_tensor_parallel_group(config.tensor_parallel_size);
    auto encoder = bart::load_engine(
        context.backend, bart::require_section(context.reader, "encoder.plan"), "bart encoder");

    ModuleCreateOptions options;
    if (config.tensor_parallel_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string decoder_section = config.tensor_parallel_size > 1
                                            ? tp_engine_section_name(group.rank)
                                            : std::string("engine.plan");
    const auto& plan = bart::require_section(context.reader, decoder_section);
    auto decoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("bart failed to load decoder");
    decoder->set_timing_label("bart decoder");
    const cudaStream_t stream = decoder->stream();
    const int32_t kv_dim = decoder_cache_row_width(*decoder, config);
    auto state = std::make_unique<BartKvCache>(config.decoder_layers, config.max_cache_length,
                                               kv_dim, stream);
    if (!state->ok())
        throw std::runtime_error("bart failed to create KV cache");

    std::string task = std::string(internal::IConditionalTextGeneration::kTask);
    if (!context.reader.info().task.empty()) {
        task = context.reader.info().task;
        if (task != internal::IConditionalTextGeneration::kTask && task != "text_generation") {
            throw std::runtime_error("bart does not implement task: " + task);
        }
    }

    return new BartPipeline(std::move(encoder), std::move(decoder), std::move(state),
                            config.hidden_size, config.decoder_layers, config.max_source_length,
                            config.decoder_start_token_id, config.eos_token_id, config.bos_token_id,
                            config.pad_token_id, config.vocab_size, stream,
                            bart::create_tokenizer(context.reader), std::move(task), std::string{});
}

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("bart does not support --kv-cache-size");
    return trtmc::create_bart(context);
}
