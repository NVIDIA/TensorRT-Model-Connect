/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nomic_bert/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::nomic_bert {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

NomicBertRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    NomicBertRuntimeConfig config;
    config.max_sequence_length = json.at("max_sequence_length").get<std::int32_t>();
    config.hidden_size = json.at("hidden_size").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    if (config.max_sequence_length < 2 || config.hidden_size <= 0 || config.pad_token_id < 0)
        throw std::runtime_error("nomic_bert runtime.json does not match its runtime contract");
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::ITextToEmbedding::kTask)
        throw std::runtime_error("nomic_bert bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::nomic_bert

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nomic_bert does not support --kv-cache-size");
    const auto config_data = trtmc::nomic_bert::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::nomic_bert::require_section(context.reader, "model.plan");
    const auto tokenizer_data =
        trtmc::nomic_bert::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::nomic_bert::parse_config(config_data);
    auto primary_task = trtmc::nomic_bert::require_task(context.reader);
    auto tokenizer = trtmc::CreateWordPieceTokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                                     /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("nomic_bert tokenizer failed to load");
    trtmc::ModuleCreateOptions options{};
    auto encoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (!encoder || !encoder->ok())
        throw std::runtime_error("nomic_bert engine failed to load");
    return new trtmc::NomicBertPipeline(std::move(encoder),
                                        std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
                                        std::move(config), std::move(primary_task));
}
