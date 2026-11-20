/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mobilebert/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::mobilebert {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

MobileBertRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    MobileBertRuntimeConfig config;
    config.max_sequence_length = json.at("max_sequence_length").get<std::int32_t>();
    config.vocab_size = json.at("vocab_size").get<std::int32_t>();
    config.type_vocab_size = json.at("type_vocab_size").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.unk_token_id = json.at("unk_token_id").get<std::int32_t>();
    config.cls_token_id = json.at("cls_token_id").get<std::int32_t>();
    config.sep_token_id = json.at("sep_token_id").get<std::int32_t>();
    config.mask_token_id = json.at("mask_token_id").get<std::int32_t>();
    config.vocabulary_id = json.at("vocabulary_id").get<std::string>();
    config.relation_labels = json.at("relation_labels").get<std::vector<std::string>>();
    config.relation_vocabulary_id = json.at("relation_vocabulary_id").get<std::string>();
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IMaskedTextToTokenScores::kTask &&
        task != internal::ITextPairToPretrainingRelationScores::kTask)
        throw std::runtime_error("mobilebert bundle declares an unsupported task: " + task);
    return task;
}

} // namespace
} // namespace trtmc::mobilebert

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("mobilebert does not support --kv-cache-size");
    const auto config_data = trtmc::mobilebert::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::mobilebert::require_section(context.reader, "model.plan");
    const auto tokenizer_json =
        trtmc::mobilebert::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::mobilebert::parse_config(config_data);
    auto primary_task = trtmc::mobilebert::require_task(context.reader);
    auto tokenizer = trtmc::CreateMobileBertTokenizer(tokenizer_json.data(), tokenizer_json.size());
    if (static_cast<std::int64_t>(tokenizer->size()) != config.vocab_size)
        throw std::runtime_error("mobilebert tokenizer.json does not match the engine vocabulary");
    if (tokenizer->id_for_token("[MASK]") != config.mask_token_id ||
        tokenizer->id_for_token("[CLS]") != config.cls_token_id ||
        tokenizer->id_for_token("[SEP]") != config.sep_token_id)
        throw std::runtime_error("mobilebert tokenizer.json disagrees with runtime.json ids");
    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("mobilebert engine failed to load");
    return new trtmc::MobileBertPipeline(
        std::move(engine), std::shared_ptr<trtmc::IMobileBertTokenizer>(std::move(tokenizer)),
        std::move(config), std::move(primary_task));
}
