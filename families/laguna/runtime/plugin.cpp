/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/kv_cache.h"
#include "families/laguna/runtime/pipeline.h"
#include "families/laguna/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::laguna {

namespace {

struct RuntimeConfig {
    std::int32_t vocab_size;
    std::int32_t hidden_size;
    std::int32_t num_layers;
    std::int32_t num_key_value_heads;
    std::int32_t head_dim;
    std::int32_t bos_token_id;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t pad_token_id;
    std::int32_t max_cache_length;
    std::int32_t prefill_chunk_rows;
    std::string precision;
    bool chat_template;
};

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("laguna bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("laguna runtime.json is missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("laguna runtime.json has an invalid '") + name + "'");
    }
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const auto data = require_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(data.begin(), data.end());
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("laguna runtime.json is invalid: ") + error.what());
    }
    if (!json.is_object() || json.size() != 12)
        throw std::runtime_error("laguna runtime.json has an unexpected field set");
    RuntimeConfig config{
        require_value<std::int32_t>(json, "vocab_size"),
        require_value<std::int32_t>(json, "hidden_size"),
        require_value<std::int32_t>(json, "num_hidden_layers"),
        require_value<std::int32_t>(json, "num_key_value_heads"),
        require_value<std::int32_t>(json, "head_dim"),
        require_value<std::int32_t>(json, "bos_token_id"),
        require_value<std::vector<std::int32_t>>(json, "eos_token_ids"),
        require_value<std::int32_t>(json, "pad_token_id"),
        require_value<std::int32_t>(json, "max_cache_length"),
        require_value<std::int32_t>(json, "prefill_chunk_rows"),
        require_value<std::string>(json, "precision"),
        false,
    };
    // "laguna" when the builder matched the checkpoint's chat template to the
    // renderer in chat_template.cpp; null otherwise.
    const auto& chat = json.contains("chat_template") ? json.at("chat_template") : nlohmann::json();
    if (chat.is_string() && chat.get<std::string>() == "laguna")
        config.chat_template = true;
    else if (!chat.is_null())
        throw std::runtime_error("laguna runtime.json has an invalid 'chat_template'");
    if (config.vocab_size <= 0 || config.hidden_size <= 0 || config.num_layers <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.max_cache_length <= 0 ||
        config.prefill_chunk_rows <= 0 || config.eos_token_ids.empty())
        throw std::runtime_error("laguna runtime.json contains invalid dimensions");
    if (config.precision != "bf16" && config.precision != "fp32")
        throw std::runtime_error("laguna runtime.json contains an invalid precision");
    return config;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config = parse_runtime_config(context.reader);
    BackendDualProfileModules modules;
    {
        const auto plan = require_section(context.reader, "engine.plan");
        modules = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    }
    if (modules.prefill == nullptr || !modules.prefill->ok() || modules.decode == nullptr ||
        !modules.decode->ok())
        throw std::runtime_error("laguna engine did not create prefill and decode contexts");
    modules.prefill->set_timing_label("laguna prefill");
    modules.decode->set_timing_label("laguna decode");

    const auto max_rows = modules.prefill->input_profile_shape(
        "token_id", modules.prefill->profile_idx(), ProfileShapeSelector::kMax);
    if (max_rows.size() != 1 || max_rows[0] != config.prefill_chunk_rows)
        throw std::runtime_error("laguna prefill profile does not match runtime.json");

    auto cache = std::make_unique<KvCache>(
        config.num_layers, config.max_cache_length, config.num_key_value_heads * config.head_dim,
        modules.decode->stream(), config.precision == "bf16" ? DType::kBFloat16 : DType::kFloat32);
    if (!cache->ok())
        throw std::runtime_error("laguna failed to allocate its KV cache");

    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    auto tokenizer = Tokenizer::create({tokenizer_json.data(), tokenizer_json.size()});

    PipelineConfig pipeline;
    pipeline.vocab_size = config.vocab_size;
    pipeline.num_layers = config.num_layers;
    pipeline.prefill_chunk_rows = config.prefill_chunk_rows;
    pipeline.eos_token_ids = config.eos_token_ids;
    pipeline.has_chat_template = config.chat_template;
    return new Pipeline(std::move(modules.prefill), std::move(modules.decode), std::move(cache),
                        std::move(tokenizer), std::move(pipeline));
}

} // namespace trtmc::laguna

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("laguna does not support --kv-cache-size");
    return trtmc::laguna::create(context);
}
