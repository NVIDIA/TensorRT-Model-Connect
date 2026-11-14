/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deberta_v2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::deberta_v2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

DebertaV2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    DebertaV2RuntimeConfig config;
    config.max_sequence_length = json.at("max_sequence_length").get<std::int32_t>();
    config.hidden_size = json.at("hidden_size").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.cls_token_id = json.at("cls_token_id").get<std::int32_t>();
    config.sep_token_id = json.at("sep_token_id").get<std::int32_t>();
    if (config.max_sequence_length < 2 || config.hidden_size <= 0 || config.pad_token_id < 0 ||
        config.cls_token_id < 0 || config.sep_token_id < 0) {
        throw std::runtime_error("deberta_v2 runtime.json does not match its runtime contract");
    }
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::ITextToTokenFeatures::kTask)
        throw std::runtime_error("deberta_v2 bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::deberta_v2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("deberta_v2 does not support --kv-cache-size");
    const auto config_data = trtmc::deberta_v2::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::deberta_v2::require_section(context.reader, "model.plan");
    const auto tokenizer_data =
        trtmc::deberta_v2::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::deberta_v2::parse_config(config_data);
    auto primary_task = trtmc::deberta_v2::require_task(context.reader);
    auto tokenizer = trtmc::CreateDebertaTokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                                   /*add_special_tokens=*/false);
    if (!tokenizer)
        throw std::runtime_error("deberta_v2 tokenizer failed to load");
    trtmc::ModuleCreateOptions options{};
    auto encoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (!encoder || !encoder->ok())
        throw std::runtime_error("deberta_v2 engine failed to load");
    return new trtmc::DebertaV2Pipeline(std::move(encoder),
                                        std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
                                        std::move(config), std::move(primary_task));
}
