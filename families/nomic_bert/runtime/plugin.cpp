/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/nomic_bert/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {

std::vector<char> section(const trtmc::BundleReader& reader, const char* name) {
    const auto* info = reader.find_section(name);
    if (!info || !info->length)
        throw std::runtime_error("missing Nomic bundle section: " + std::string(name));
    return reader.read_section(name);
}

int32_t integer(const nlohmann::json& config, const char* name, int32_t low, int32_t high) {
    const auto& value = config.at(name);
    if (!value.is_number_integer())
        throw std::invalid_argument("Nomic runtime field must be an integer: " + std::string(name));
    const auto number = value.get<int64_t>();
    if (number < low || number > high)
        throw std::invalid_argument("invalid Nomic runtime field: " + std::string(name));
    return static_cast<int32_t>(number);
}

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes)
        throw std::invalid_argument("nomic_bert does not support --kv-cache-size");
    if (context.reader.info().task != "text_to_embedding")
        throw std::invalid_argument("Nomic supports only text_to_embedding");
    const auto data = section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(data.begin(), data.end());
    const auto length = integer(config, "max_sequence_length", 2, 2048);
    const auto vocab = integer(config, "vocab_size", 103, 1000000);
    integer(config, "hidden_size", 768, 768);
    const auto embedding_space = config.at("embedding_space").get<std::string>();
    auto tokenizer_data = section(context.reader, "tokenizer.json");
    auto tokenizer =
        trtmc::nomic_bert::CreateWordPieceTokenizer(tokenizer_data.data(), tokenizer_data.size());
    const auto plan = section(context.reader, "engine.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    return new trtmc::nomic_bert::Pipeline(std::move(engine), std::move(tokenizer), length, vocab,
                                           embedding_space);
}
