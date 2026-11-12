/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/apertus/runtime/model.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::apertus {
namespace {

struct RuntimeConfig {
    std::int32_t vocab_size;
    std::int32_t num_layers;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t max_cache_length;
    std::int32_t prefill_chunk_tokens;
    std::string precision;
    std::vector<std::int32_t> eos_token_ids;
    std::vector<std::int32_t> prefix_ids;
    std::vector<std::int32_t> suffix_ids;
};

const std::set<std::string> kRuntimeFields{
    "vocab_size",
    "num_hidden_layers",
    "num_key_value_heads",
    "head_dim",
    "max_cache_length",
    "prefill_chunk_tokens",
    "precision",
    "eos_token_ids",
    "tokenizer_add_special_tokens",
    "tokenizer_prefix_ids",
    "tokenizer_suffix_ids",
};

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("Apertus bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

template <typename T>
T require(const nlohmann::json& json, const char* name) {
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("Apertus runtime.json has an invalid '") + name + "'");
    }
}

std::int32_t positive(const nlohmann::json& json, const char* name) {
    const auto value = require<std::int32_t>(json, name);
    if (value <= 0)
        throw std::runtime_error(std::string("Apertus runtime.json '") + name +
                                 "' must be positive");
    return value;
}

RuntimeConfig parse_runtime_config(const BundleReader& bundle) {
    const auto text = require_section(bundle, "runtime.json");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text.begin(), text.end());
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Apertus runtime.json is not JSON: ") + error.what());
    }
    if (!json.is_object())
        throw std::runtime_error("Apertus runtime.json must be an object");
    std::set<std::string> fields;
    for (const auto& item : json.items())
        fields.insert(item.key());
    if (fields != kRuntimeFields)
        throw std::runtime_error("Apertus runtime.json has an unexpected field set");
    RuntimeConfig config{
        positive(json, "vocab_size"),
        positive(json, "num_hidden_layers"),
        positive(json, "num_key_value_heads"),
        positive(json, "head_dim"),
        positive(json, "max_cache_length"),
        positive(json, "prefill_chunk_tokens"),
        require<std::string>(json, "precision"),
        require<std::vector<std::int32_t>>(json, "eos_token_ids"),
        require<std::vector<std::int32_t>>(json, "tokenizer_prefix_ids"),
        require<std::vector<std::int32_t>>(json, "tokenizer_suffix_ids"),
    };
    if (!require<bool>(json, "tokenizer_add_special_tokens"))
        throw std::runtime_error(
            "Apertus text prefixes always receive the tokenizer's special tokens");
    if (config.precision != "bf16" && config.precision != "fp32")
        throw std::runtime_error("Apertus runtime.json precision must be bf16 or fp32");
    if (config.prefill_chunk_tokens > config.max_cache_length || config.eos_token_ids.empty())
        throw std::runtime_error("Apertus runtime.json has an invalid generation contract");
    for (const auto* ids : {&config.eos_token_ids, &config.prefix_ids, &config.suffix_ids}) {
        for (const auto id : *ids) {
            if (id < 0 || id >= config.vocab_size)
                throw std::runtime_error("Apertus runtime.json token ID is outside the vocabulary");
        }
    }
    return config;
}

std::set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> result;
    for (const auto& tensor : tensors)
        result.insert(tensor.name);
    return result;
}

void validate_engine(const RuntimeConfig& config, const ITrtModule& module,
                     std::int64_t max_tokens) {
    std::set<std::string> inputs{"token_id", "position_id", "cache_write_indices",
                                 "key_value_lengths", "key_positions"};
    std::set<std::string> outputs{"logits"};
    for (std::int32_t layer = 0; layer < config.num_layers; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        inputs.insert({"cache_k" + suffix, "cache_v" + suffix});
        outputs.insert({"present_k" + suffix, "present_v" + suffix});
    }
    if (names(module.input_info()) != inputs || names(module.output_info()) != outputs)
        throw std::runtime_error("Apertus engine tensor inventory is not exact");
    if (module.optimization_profile_count() != 2)
        throw std::runtime_error("Apertus engine must have prefill and decode profiles");
    for (const char* name : {"token_id", "position_id"}) {
        if (module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMin) !=
                std::vector<int64_t>{1} ||
            module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax) !=
                std::vector<int64_t>{max_tokens}) {
            throw std::runtime_error(std::string("Apertus engine profile mismatch for '") + name +
                                     "'");
        }
    }
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (std::string(context.backend.name()) != "trt")
        throw std::runtime_error("Apertus requires the TensorRT backend");
    const auto config = parse_runtime_config(context.reader);
    const auto plan = require_section(context.reader, "engine.plan");
    auto modules = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (modules.prefill == nullptr || !modules.prefill->ok() || modules.decode == nullptr ||
        !modules.decode->ok()) {
        throw std::runtime_error("Apertus engine did not create prefill and decode contexts");
    }
    validate_engine(config, *modules.prefill, config.prefill_chunk_tokens);
    validate_engine(config, *modules.decode, 1);
    const auto dtype = config.precision == "bf16" ? DType::kBFloat16 : DType::kFloat32;
    auto cache =
        std::make_unique<KvCache>(config.num_layers, config.num_kv_heads, config.head_dim,
                                  config.max_cache_length, dtype, modules.decode->stream());
    if (!cache->ok())
        throw std::runtime_error("Apertus KV cache allocation failed");
    cache->bind(*modules.prefill);
    cache->bind(*modules.decode);
    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    auto tokenizer =
        std::make_unique<Tokenizer>(std::string_view(tokenizer_json.data(), tokenizer_json.size()));
    if (tokenizer->vocab_size() > config.vocab_size)
        throw std::runtime_error("Apertus tokenizer has IDs outside the model vocabulary");
    ModelConfig model_config{config.vocab_size, config.prefill_chunk_tokens, config.eos_token_ids,
                             config.prefix_ids, config.suffix_ids};
    return new ApertusModel(std::move(modules.prefill), std::move(modules.decode), std::move(cache),
                            std::move(tokenizer), std::move(model_config));
}

} // namespace trtmc::apertus

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("apertus does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::ITextContinuation::kTask)
        throw std::invalid_argument("apertus bundles serve only text_continuation");
    return trtmc::apertus::create(context);
}
