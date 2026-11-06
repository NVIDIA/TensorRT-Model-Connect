/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/plugin_helpers.h"

#include "families/lfm3/runtime/byte_level_decoder.h"
#include "families/lfm3/runtime/chat_templates.h"
#include "families/lfm3/runtime/pretokenizer.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trtmc::lfm3 {

namespace {

std::string chat_template_source(const BundleReader& bundle) {
    if (const auto* config = bundle.find_section("tokenizer_config.json");
        config != nullptr && config->length > 0) {
        try {
            const auto data = bundle.read_section("tokenizer_config.json");
            const auto json = nlohmann::json::parse(data.begin(), data.end());
            if (json.is_object() && json.contains("chat_template") &&
                json.at("chat_template").is_string()) {
                return json.at("chat_template").get<std::string>();
            }
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error("lfm3 invalid tokenizer_config.json: " +
                                     std::string(error.what()));
        }
    }
    if (const auto* source = bundle.find_section("chat_template.jinja");
        source != nullptr && source->length > 0) {
        const auto data = bundle.read_section("chat_template.jinja");
        return {data.begin(), data.end()};
    }
    return {};
}

} // namespace

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto& section = require_section(bundle, name);
    return {section.begin(), section.end()};
}

std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan) {
    auto module = backend.create_module(plan.data(), plan.size(), {});
    if (module == nullptr || !module->ok())
        throw std::runtime_error("lfm3 failed to load engine.plan");
    module->set_timing_label("lfm3 engine");
    return module;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& section = require_section(bundle, "tokenizer.json");
    const bool pinned =
        lfm3_uses_pinned_split_byte_level_pretokenizer(section.data(), section.size());
    auto tokenizer = CreateBpeTokenizer(section.data(), section.size(), !pinned);
    if (tokenizer == nullptr)
        throw std::runtime_error("lfm3 BPE tokenizer construction failed");
    if (lfm3_uses_sequence_byte_level_decoder(section.data(), section.size()))
        tokenizer = lfm3_wrap_byte_level_decoder(std::move(tokenizer));
    if (pinned) {
        tokenizer = lfm3_wrap_pinned_pretokenizer(
            std::move(tokenizer), lfm3_tokenizer_added_tokens(section.data(), section.size()));
    }
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

DType state_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    throw std::runtime_error("lfm3 requires fp16 or bf16 state precision");
}

Lfm3KvCacheNames kv_names(std::int32_t num_attention_layers) {
    Lfm3KvCacheNames names;
    for (std::int32_t layer = 0; layer < num_attention_layers; ++layer) {
        const std::string suffix = std::to_string(layer);
        names.cache_k.push_back("cache_k_" + suffix);
        names.cache_v.push_back("cache_v_" + suffix);
        names.present_k.push_back("present_k_" + suffix);
        names.present_v.push_back("present_v_" + suffix);
    }
    return names;
}

void apply_chat_template(const BundleReader& bundle, Lfm3TextGenConfig& config) {
    config.chat_template_format = lfm3_detect_chat_template_format(chat_template_source(bundle));
}

} // namespace trtmc::lfm3
