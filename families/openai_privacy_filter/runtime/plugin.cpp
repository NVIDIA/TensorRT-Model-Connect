/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/openai_privacy_filter/runtime/o200k_tokenizer.h"
#include "families/openai_privacy_filter/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::openai_privacy_filter {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::int64_t positive(const nlohmann::json& config, const char* name) {
    if (!config.contains(name) || !config.at(name).is_number_integer())
        throw std::runtime_error("openai_privacy_filter runtime.json requires integer " +
                                 std::string(name));
    const auto value = config.at(name).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("openai_privacy_filter runtime.json has invalid " +
                                 std::string(name));
    return value;
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto config = nlohmann::json::parse(data.begin(), data.end());
    if (!config.is_object() || !config.contains("labels") || !config.at("labels").is_array() ||
        config.at("labels").empty())
        throw std::runtime_error("openai_privacy_filter runtime.json requires its label names");
    return {positive(config, "vocab_size"), positive(config, "max_sequence_length"),
            static_cast<std::int64_t>(config.at("labels").size())};
}

} // namespace
} // namespace trtmc::openai_privacy_filter

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace family = trtmc::openai_privacy_filter;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("openai_privacy_filter does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ITextToHeadScores::kTask)
        throw std::runtime_error("openai_privacy_filter does not implement task: " + task);
    const auto config =
        family::parse_config(family::require_section(context.reader, "runtime.json"));
    const auto tokenizer_json = family::require_section(context.reader, "tokenizer.json");
    auto tokenizer = std::make_shared<family::O200kTokenizer>(family::O200kTokenizer::from_json(
        std::string_view(tokenizer_json.data(), tokenizer_json.size())));
    if (tokenizer->vocabulary_size() > static_cast<std::size_t>(config.vocab_size))
        throw std::runtime_error("openai_privacy_filter tokenizer exceeds the embedding table");
    const auto plan = family::require_section(context.reader, "engine.plan");
    auto engine =
        context.backend.create_module(plan.data(), plan.size(), trtmc::ModuleCreateOptions{});
    if (!engine || !engine->ok())
        throw std::runtime_error("openai_privacy_filter engine failed to load");
    return new family::TokenClassifier(
        std::move(engine), [tokenizer](std::string_view text) { return tokenizer->encode(text); },
        config);
}
