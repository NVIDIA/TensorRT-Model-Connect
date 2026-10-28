/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esm/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::esm {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

EsmRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    EsmRuntimeConfig config;
    config.max_sequence_length = json.at("max_sequence_length").get<std::int32_t>();
    config.vocab_size = json.at("vocab_size").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.cls_token_id = json.at("cls_token_id").get<std::int32_t>();
    config.eos_token_id = json.at("eos_token_id").get<std::int32_t>();
    config.mask_token_id = json.at("mask_token_id").get<std::int32_t>();
    config.unk_token_id = json.at("unk_token_id").get<std::int32_t>();
    config.vocabulary_id = json.at("vocabulary_id").get<std::string>();
    if (config.max_sequence_length < 2 || config.vocab_size <= 0 || config.pad_token_id < 0 ||
        config.cls_token_id < 0 || config.eos_token_id < 0 ||
        config.mask_token_id >= config.vocab_size || config.mask_token_id < 0 ||
        config.unk_token_id < 0 || config.vocabulary_id.empty()) {
        throw std::runtime_error("esm runtime.json does not match its runtime contract");
    }
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IMaskedTextToTokenScores::kTask)
        throw std::runtime_error("esm bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::esm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("esm does not support --kv-cache-size");
    const auto config_data = trtmc::esm::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::esm::require_section(context.reader, "model.plan");
    const auto vocabulary = trtmc::esm::require_section(context.reader, "vocab.txt");

    auto config = trtmc::esm::parse_config(config_data);
    auto primary_task = trtmc::esm::require_task(context.reader);
    auto tokenizer = trtmc::CreateEsmTokenizer(vocabulary.data(), vocabulary.size());
    if (!tokenizer)
        throw std::runtime_error("esm tokenizer failed to load");
    if (static_cast<std::int32_t>(tokenizer->size()) != config.vocab_size)
        throw std::runtime_error("esm vocab.txt does not match the engine's vocabulary width");
    trtmc::ModuleCreateOptions options{};
    auto encoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (!encoder || !encoder->ok())
        throw std::runtime_error("esm engine failed to load");
    return new trtmc::EsmPipeline(std::move(encoder),
                                  std::shared_ptr<trtmc::IEsmTokenizer>(std::move(tokenizer)),
                                  std::move(config), std::move(primary_task));
}
