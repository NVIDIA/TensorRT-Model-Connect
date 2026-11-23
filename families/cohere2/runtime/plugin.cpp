/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere2/runtime/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace trtmc::cohere2 {
namespace {
std::vector<char> section(const BundleReader& bundle, const char* name) {
    const auto* entry = bundle.find_section(name);
    if (!entry || entry->length == 0)
        throw std::invalid_argument(std::string("cohere2 missing bundle section: ") + name);
    return bundle.read_section(name);
}
std::set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> result;
    for (const auto& item : tensors)
        result.insert(item.name);
    return result;
}
} // namespace

ITask* create(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("cohere2 does not support --kv-cache-size");
    if (context.reader.info().task != internal::ITextContinuation::kTask ||
        std::string(context.backend.name()) != "trt")
        throw std::invalid_argument("cohere2 requires text_continuation");
    const auto raw = section(context.reader, "runtime.json");
    const auto data = nlohmann::json::parse(raw.begin(), raw.end());
    const std::set<std::string> expected{"max_sequence_length", "vocab_size",   "bos_token_id",
                                         "eos_token_id",        "pad_token_id", "precision"};
    std::set<std::string> fields;
    for (const auto& item : data.items())
        fields.insert(item.key());
    if (fields != expected || data.at("precision") != "fp32")
        throw std::invalid_argument("cohere2 invalid runtime contract");
    for (const auto* key :
         {"max_sequence_length", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"}) {
        if (!data.at(key).is_number_integer())
            throw std::invalid_argument(
                "cohere2 runtime dimensions and token IDs must be integers");
    }
    ModelConfig config{
        data.at("max_sequence_length").get<std::int32_t>(),
        data.at("vocab_size").get<std::int32_t>(), data.at("bos_token_id").get<std::int32_t>(),
        data.at("eos_token_id").get<std::int32_t>(), data.at("pad_token_id").get<std::int32_t>()};
    if (config.length <= 0 || config.vocabulary <= 0) {
        throw std::invalid_argument("cohere2 invalid runtime dimensions");
    }
    for (const auto id : {config.bos, config.eos, config.pad}) {
        if (id < 0 || id >= config.vocabulary)
            throw std::invalid_argument("cohere2 special token is outside vocabulary");
    }
    const auto plan = section(context.reader, "engine.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), ModuleCreateOptions{});
    if (!engine || !engine->ok() ||
        names(engine->input_info()) != std::set<std::string>{"token_ids", "last_position"} ||
        names(engine->output_info()) != std::set<std::string>{"logits"} ||
        engine->tensor_shape("token_ids") != std::vector<std::int64_t>{config.length} ||
        engine->tensor_shape("last_position") != std::vector<std::int64_t>{1} ||
        engine->tensor_shape("logits") != std::vector<std::int64_t>{1, config.vocabulary} ||
        engine->tensor_dtype("token_ids") != DType::kInt32 ||
        engine->tensor_dtype("last_position") != DType::kInt32 ||
        engine->tensor_dtype("logits") != DType::kFloat32)
        throw std::invalid_argument("cohere2 engine does not match its runtime contract");
    const auto serialized = section(context.reader, "tokenizer.json");
    auto tokenizer =
        std::make_unique<Tokenizer>(std::string_view(serialized.data(), serialized.size()));
    return new Model(std::move(engine), std::move(tokenizer), config);
}
} // namespace trtmc::cohere2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::cohere2::create(context);
}
