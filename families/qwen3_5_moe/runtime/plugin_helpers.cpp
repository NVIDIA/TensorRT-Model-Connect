/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5_moe/runtime/plugin_helpers.h"

#include <charconv>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::qwen3_5_moe {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto& data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label) {
    auto engine = backend.create_module(plan.data(), plan.size(), {});
    if (engine == nullptr || !engine->ok())
        throw std::runtime_error(std::string("qwen3_5_moe failed to load ") + label);
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    auto data = require_section(bundle, "tokenizer.json");
    // Base checkpoints can define additional tokens only in tokenizer_config.json.
    // Preserve their source IDs instead of splitting their spelling into ordinary BPE tokens.
    if (bundle.find_section("tokenizer_config.json")) {
        const auto config =
            nlohmann::json::parse(require_text_section(bundle, "tokenizer_config.json"));
        if (config.contains("added_tokens_decoder")) {
            const auto& extra = config.at("added_tokens_decoder");
            if (!extra.is_object())
                throw std::runtime_error("Qwen3.5-MoE added_tokens_decoder must be an object");
            auto tokenizer_json = nlohmann::json::parse(data);
            if (!tokenizer_json.contains("added_tokens"))
                tokenizer_json["added_tokens"] = nlohmann::json::array();
            auto& added = tokenizer_json.at("added_tokens");
            if (!added.is_array())
                throw std::runtime_error("Qwen3.5-MoE added_tokens must be an array");
            for (const auto& entry : extra.items()) {
                int32_t id = -1;
                const auto& key = entry.key();
                const auto parsed = std::from_chars(key.data(), key.data() + key.size(), id);
                if (parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() || id < 0)
                    throw std::runtime_error(
                        "Qwen3.5-MoE added token ID must be a nonnegative int32");
                const auto& definition = entry.value();
                const auto content = definition.at("content").get<std::string>();
                bool present = false;
                for (const auto& token : added) {
                    const auto token_id = token.at("id").get<int32_t>();
                    const auto token_content = token.at("content").get<std::string>();
                    if (token_id == id || token_content == content) {
                        if (token_id != id || token_content != content)
                            throw std::runtime_error(
                                "Qwen3.5-MoE tokenizer files disagree on an added token");
                        for (const auto* flag :
                             {"single_word", "lstrip", "rstrip", "normalized", "special"}) {
                            if (token.value(flag, false) != definition.value(flag, false))
                                throw std::runtime_error(
                                    "Qwen3.5-MoE tokenizer files disagree on added-token behavior");
                        }
                        present = true;
                        break;
                    }
                }
                if (!present) {
                    auto token = definition;
                    token["id"] = id;
                    added.push_back(std::move(token));
                }
            }
            const auto serialized = tokenizer_json.dump();
            data.assign(serialized.begin(), serialized.end());
        }
    }
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (!tokenizer)
        throw std::runtime_error("qwen3_5_moe BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}
} // namespace trtmc::qwen3_5_moe
