/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/engine_contract.h"
#include "families/openai_gpt/runtime/model.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace trtmc::openai_gpt {
namespace {

struct RuntimeConfig {
    std::int32_t vocab_size;
    std::int32_t num_layers;
    std::int32_t num_heads;
    std::int32_t head_dim;
    std::int32_t max_cache_length;
    std::vector<std::int32_t> eos_token_ids;
};

const std::set<std::string> kRuntimeFields{
    "vocab_size",       "num_hidden_layers", "num_attention_heads", "head_dim",
    "max_cache_length", "precision",         "eos_token_ids",
};

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("OpenAI GPT bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

// A JSON integer inside [low, high]; checked before the conversion, so a fraction or an
// out-of-range value is refused rather than truncated or wrapped.
std::int32_t integer_in(const nlohmann::json& value, std::int64_t low, const char* name) {
    const bool fits = (value.is_number_unsigned() &&
                       value.get<std::uint64_t>() <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) &&
                       static_cast<std::int64_t>(value.get<std::uint64_t>()) >= low) ||
                      (value.is_number_integer() && !value.is_number_unsigned() &&
                       value.get<std::int64_t>() >= low &&
                       value.get<std::int64_t>() <= std::numeric_limits<std::int32_t>::max());
    if (!fits)
        throw std::runtime_error(std::string("OpenAI GPT runtime.json '") + name +
                                 "' must be an integer >= " + std::to_string(low));
    return static_cast<std::int32_t>(value.get<std::int64_t>());
}

nlohmann::json parse_unique_object(const std::vector<char>& text) {
    std::vector<std::unordered_set<std::string>> keys;
    const auto reject_repeated_keys = [&](int, nlohmann::json::parse_event_t event,
                                          nlohmann::json& parsed) {
        if (event == nlohmann::json::parse_event_t::object_start)
            keys.emplace_back();
        else if (event == nlohmann::json::parse_event_t::object_end)
            keys.pop_back();
        else if (event == nlohmann::json::parse_event_t::key &&
                 !keys.back().insert(parsed.get<std::string>()).second)
            throw std::runtime_error("OpenAI GPT runtime.json repeats the key '" +
                                     parsed.get<std::string>() + "'");
        return true;
    };
    try {
        return nlohmann::json::parse(text.begin(), text.end(), reject_repeated_keys);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("OpenAI GPT runtime.json is not JSON: ") +
                                 error.what());
    }
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const auto json = parse_unique_object(require_section(bundle, "runtime.json"));
    if (!json.is_object())
        throw std::runtime_error("OpenAI GPT runtime.json must be an object");
    std::set<std::string> fields;
    for (const auto& item : json.items())
        fields.insert(item.key());
    if (fields != kRuntimeFields)
        throw std::runtime_error("OpenAI GPT runtime.json has an unexpected field set");
    if (!json.at("precision").is_string() || json.at("precision").get<std::string>() != "fp32")
        throw std::runtime_error("OpenAI GPT runtime.json precision must be fp32");
    RuntimeConfig config{
        integer_in(json.at("vocab_size"), 1, "vocab_size"),
        integer_in(json.at("num_hidden_layers"), 1, "num_hidden_layers"),
        integer_in(json.at("num_attention_heads"), 1, "num_attention_heads"),
        integer_in(json.at("head_dim"), 1, "head_dim"),
        integer_in(json.at("max_cache_length"), 1, "max_cache_length"),
        {},
    };
    const auto& eos = json.at("eos_token_ids");
    if (!eos.is_array())
        throw std::runtime_error("OpenAI GPT runtime.json 'eos_token_ids' must be a list");
    for (const auto& value : eos) {
        const auto id = integer_in(value, 0, "eos_token_ids");
        if (id >= config.vocab_size)
            throw std::runtime_error("OpenAI GPT runtime.json EOS id is outside the vocabulary");
        config.eos_token_ids.push_back(id);
    }
    return config;
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (std::string(context.backend.name()) != "trt")
        throw std::runtime_error("OpenAI GPT requires the TensorRT backend");
    const auto config = parse_runtime_config(context.reader);
    const auto plan = require_section(context.reader, "engine.plan");
    auto modules = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (modules.prefill == nullptr || !modules.prefill->ok() || modules.decode == nullptr ||
        !modules.decode->ok()) {
        throw std::runtime_error("OpenAI GPT engine did not create prefill and decode contexts");
    }
    const EngineShape shape{config.vocab_size, config.num_layers, config.num_heads, config.head_dim,
                            config.max_cache_length};
    validate_engine(shape, *modules.prefill, config.max_cache_length);
    validate_engine(shape, *modules.decode, 1);
    auto cache = std::make_unique<KvCache>(config.num_layers, config.num_heads, config.head_dim,
                                           config.max_cache_length, modules.decode->stream());
    if (!cache->ok())
        throw std::runtime_error("OpenAI GPT KV cache allocation failed");
    cache->bind(*modules.prefill);
    cache->bind(*modules.decode);
    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    auto tokenizer =
        std::make_unique<Tokenizer>(std::string_view(tokenizer_json.data(), tokenizer_json.size()));
    // As the builder requires: every id the engine can produce has a token.
    if (tokenizer->vocab_size() != config.vocab_size)
        throw std::runtime_error("OpenAI GPT tokenizer vocabulary differs from the model's");
    return new OpenAIGPTModel(std::move(modules.prefill), std::move(modules.decode),
                              std::move(cache), std::move(tokenizer),
                              ModelConfig{config.vocab_size, config.eos_token_ids});
}

} // namespace trtmc::openai_gpt

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("openai_gpt does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::ITextContinuation::kTask)
        throw std::invalid_argument("openai_gpt bundles serve only text_continuation");
    return trtmc::openai_gpt::create(context);
}
