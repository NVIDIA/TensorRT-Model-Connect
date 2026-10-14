/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2_audio/runtime/plugin_helpers.h"

#include "families/lfm2_audio/runtime/byte_level_decoder.h"
#include "families/lfm2_audio/runtime/pretokenizer.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trtmc::lfm2_audio {

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
        throw std::runtime_error("lfm2 failed to load engine.plan");
    module->set_timing_label("lfm2 engine");
    return module;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& section = require_section(bundle, "tokenizer.json");
    const bool pinned =
        lfm2_audio_uses_pinned_split_byte_level_pretokenizer(section.data(), section.size());
    if (!pinned)
        throw std::runtime_error("lfm2_audio tokenizer.json has an unsupported pre_tokenizer");
    auto tokenizer = CreateBpeTokenizer(section.data(), section.size(), false);
    if (tokenizer == nullptr)
        throw std::runtime_error("lfm2 BPE tokenizer construction failed");
    if (lfm2_audio_uses_sequence_byte_level_decoder(section.data(), section.size()))
        tokenizer = lfm2_audio_wrap_byte_level_decoder(std::move(tokenizer));
    tokenizer = lfm2_audio_wrap_pinned_pretokenizer(
        std::move(tokenizer), lfm2_audio_tokenizer_added_tokens(section.data(), section.size()));
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

DType state_dtype(const std::string& precision) {
    if (precision == "fp32")
        return DType::kFloat32;
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    throw std::runtime_error("lfm2 requires fp16 or bf16 state precision");
}

Lfm2AudioKvCacheNames kv_names(std::int32_t num_attention_layers) {
    Lfm2AudioKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_attention_layers; ++layer) {
        const std::string suffix = std::to_string(layer);
        names.cache_k.push_back("cache_k_" + suffix);
        names.cache_v.push_back("cache_v_" + suffix);
        names.present_k.push_back("present_k_" + suffix);
        names.present_v.push_back("present_v_" + suffix);
    }
    return names;
}

} // namespace trtmc::lfm2_audio
