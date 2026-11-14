/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/embeddinggemma2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
std::vector<char> section(const trtmc::BundleReader& reader, const char* name) {
    const auto* entry = reader.find_section(name);
    if (!entry || entry->length == 0)
        throw std::runtime_error(std::string("EmbeddingGemma2 missing section: ") + name);
    return reader.read_section(name);
}
int32_t positive(const nlohmann::json& config, const char* key) {
    const auto& value = config.at(key);
    if (!value.is_number_integer() || value.get<int64_t>() < 1 || value.get<int64_t>() > INT32_MAX)
        throw std::invalid_argument(std::string("EmbeddingGemma2 invalid ") + key);
    return value.get<int32_t>();
}
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("embeddinggemma2 does not support --kv-cache-size");
    const auto& info = context.reader.info();
    if (info.family != "embeddinggemma2" || info.task != "embedding")
        throw std::invalid_argument("EmbeddingGemma2 bundle identity mismatch");
    const auto bytes = section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (config.at("format_version") != 1 || config.at("pooling") != "mean" ||
        config.at("normalization") != "l2")
        throw std::invalid_argument("EmbeddingGemma2 pooling contract mismatch");
    const auto dimension = positive(config, "embedding_dimension");
    const auto vocab = positive(config, "vocab_size");
    const auto capacity = positive(config, "max_sequence_length");
    const auto tokenizer_data = section(context.reader, "tokenizer.json");
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(), true);
    const auto plan = section(context.reader, "engine.plan");
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    return new trtmc::EmbeddingGemma2Pipeline(
        std::move(module), std::move(tokenizer), dimension, vocab, capacity,
        config.at("unsupported_token_ids").get<std::vector<int32_t>>(),
        config.at("model_id").get<std::string>());
}
